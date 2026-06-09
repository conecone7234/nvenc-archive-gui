from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    import psutil
except ImportError:
    psutil = None

try:
    from ffmpeg_nvenc_gui.core import (
        AppPaths,
        EncodeSettings,
        FileStatus,
        JobSpec,
        build_ffmpeg_command,
        build_job_specs,
        build_paths,
        clear_state,
        command_to_text,
        ensure_dirs,
        load_state,
        output_path_for,
        resumable_specs,
        save_state,
        scan_obs_files,
        temp_path_for,
    )
    from ffmpeg_nvenc_gui.ffmpeg_downloader import FfmpegDownloadError, ensure_ffmpeg_available
except ModuleNotFoundError:
    from core import (  # type: ignore
        AppPaths,
        EncodeSettings,
        FileStatus,
        JobSpec,
        build_ffmpeg_command,
        build_job_specs,
        build_paths,
        clear_state,
        command_to_text,
        ensure_dirs,
        load_state,
        output_path_for,
        resumable_specs,
        save_state,
        scan_obs_files,
        temp_path_for,
    )
    from ffmpeg_downloader import FfmpegDownloadError, ensure_ffmpeg_available  # type: ignore


@dataclass
class RuntimeJob:
    job_id: int
    spec: JobSpec
    tmp_out: Path
    out_file: Path
    log_file: Path
    command: List[str]
    process: Optional[subprocess.Popen] = None
    status: str = "waiting"


class EncoderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FFmpeg NVENC GUI Encoder")
        self.root.geometry("1280x840")

        self.paths: AppPaths = build_paths()
        ensure_dirs(self.paths)

        self.files: List[FileStatus] = []
        self.job_counter = 0
        self.pending_jobs: queue.Queue[RuntimeJob] = queue.Queue()
        self.active_jobs: Dict[int, RuntimeJob] = {}
        self.all_jobs: Dict[int, RuntimeJob] = {}

        self.lock = threading.Lock()
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.running = False
        self.paused = False
        self.stop_requested = False
        self.scheduler_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._poll_log_queue()
        self.scan_files()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        path_frame = ttk.LabelFrame(main, text="Paths", padding=8)
        path_frame.pack(fill=tk.X)

        self.path_text = tk.StringVar(
            value=(
                f"Base: {self.paths.base_dir}\n"
                f"FFmpeg: {self.paths.ffmpeg_path}\n"
                f"OBSData: {self.paths.obs_dir}"
            )
        )
        ttk.Label(path_frame, textvariable=self.path_text, justify=tk.LEFT).pack(anchor=tk.W)

        settings_frame = ttk.LabelFrame(main, text="Encode Settings", padding=8)
        settings_frame.pack(fill=tk.X, pady=(8, 0))

        self.max_jobs_var = tk.IntVar(value=2)
        self.make_4k_var = tk.BooleanVar(value=True)
        self.make_mp4_var = tk.BooleanVar(value=True)
        self.auto_download_ffmpeg_var = tk.BooleanVar(value=True)
        self.height_4k_var = tk.StringVar(value="2160")
        self.codec_var = tk.StringVar(value="hevc_nvenc")
        self.preset_var = tk.StringVar(value="p7")
        self.tune_var = tk.StringVar(value="hq")
        self.rate_mode_var = tk.StringVar(value="CQ")
        self.cq_var = tk.StringVar(value="15")
        self.bitrate_var = tk.StringVar(value="35000k")
        self.maxrate_var = tk.StringVar(value="50000k")
        self.bufsize_var = tk.StringVar(value="100000k")

        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X)

        ttk.Label(row1, text="Max jobs").pack(side=tk.LEFT)
        ttk.Spinbox(row1, from_=1, to=8, textvariable=self.max_jobs_var, width=5).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Checkbutton(row1, text="Make 4K", variable=self.make_4k_var).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Checkbutton(row1, text="Make MP4", variable=self.make_mp4_var).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Checkbutton(row1, text="Auto download FFmpeg", variable=self.auto_download_ffmpeg_var).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(row1, text="4K height").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.height_4k_var, width=8).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(row1, text="Codec").pack(side=tk.LEFT)
        ttk.Combobox(row1, textvariable=self.codec_var, values=["hevc_nvenc", "h264_nvenc", "av1_nvenc"], width=14, state="readonly").pack(side=tk.LEFT, padx=(4, 14))

        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=(8, 0))

        ttk.Label(row2, text="Preset").pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.preset_var, values=["p1", "p2", "p3", "p4", "p5", "p6", "p7"], width=6, state="readonly").pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(row2, text="Tune").pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.tune_var, values=["none", "hq", "ll", "ull", "lossless"], width=10, state="readonly").pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(row2, text="Rate mode").pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.rate_mode_var, values=["CQ", "VBR", "ABR", "CBR"], width=8, state="readonly").pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(row2, text="CQ").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.cq_var, width=8).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(row2, text="Bitrate").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.bitrate_var, width=10).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(row2, text="Maxrate").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.maxrate_var, width=10).pack(side=tk.LEFT, padx=(4, 14))
        ttk.Label(row2, text="Bufsize").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.bufsize_var, width=10).pack(side=tk.LEFT, padx=(4, 14))

        button_frame = ttk.Frame(main)
        button_frame.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(button_frame, text="Scan", command=self.scan_files).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Select All", command=self.select_all_files).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_frame, text="Clear Select", command=self.clear_selection).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_frame, text="Download FFmpeg", command=self.download_ffmpeg_button).pack(side=tk.LEFT, padx=(24, 0))
        ttk.Button(button_frame, text="Start", command=self.start_selected).pack(side=tk.LEFT, padx=(24, 0))
        ttk.Button(button_frame, text="Pause", command=self.pause_jobs).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_frame, text="Resume", command=self.resume_jobs).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_frame, text="Stop", command=self.stop_all).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_frame, text="Resume Saved", command=self.resume_saved).pack(side=tk.LEFT, padx=(24, 0))
        ttk.Button(button_frame, text="Open Log Dir", command=self.open_log_dir).pack(side=tk.LEFT, padx=(8, 0))

        body = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        list_frame = ttk.LabelFrame(body, text="OBSData Files", padding=8)
        body.add(list_frame, weight=1)

        self.file_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.config(yscrollcommand=list_scroll.set)

        right_frame = ttk.Frame(body)
        body.add(right_frame, weight=2)

        status_frame = ttk.LabelFrame(right_frame, text="Job Status", padding=8)
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)

        log_frame = ttk.LabelFrame(right_frame, text="Run Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, height=25)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{stamp}] {text}")

    def _poll_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, line + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass

        with self.lock:
            active = len(self.active_jobs)
            pending = self.pending_jobs.qsize()
            state = "Paused" if self.paused else ("Running" if self.running else "Idle")
        self.status_var.set(f"State: {state} / Active: {active} / Pending: {pending}")
        self.root.after(200, self._poll_log_queue)

    def scan_files(self) -> None:
        self.files = scan_obs_files(self.paths)
        self.file_list.delete(0, tk.END)
        for item in self.files:
            self.file_list.insert(tk.END, f"{item.label} {item.path.name}")
        self.log(f"Scanned {len(self.files)} file(s).")

    def select_all_files(self) -> None:
        self.file_list.select_set(0, tk.END)

    def clear_selection(self) -> None:
        self.file_list.selection_clear(0, tk.END)

    def open_log_dir(self) -> None:
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(self.paths.log_dir)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def download_ffmpeg_button(self) -> None:
        thread = threading.Thread(target=self._download_ffmpeg_worker, daemon=True)
        thread.start()

    def _download_ffmpeg_worker(self) -> None:
        try:
            ensure_ffmpeg_available(self.paths, auto_download=True, progress=self.log)
            self.log("FFmpeg is ready.")
        except Exception as exc:
            self.log(f"FFmpeg download failed: {exc}")
            self.root.after(0, lambda: messagebox.showerror("FFmpeg download failed", str(exc)))

    def ensure_ffmpeg_before_run(self) -> bool:
        if self.paths.ffmpeg_path.exists():
            return True

        if not self.auto_download_ffmpeg_var.get():
            messagebox.showerror("Error", f"ffmpeg.exe was not found:\n{self.paths.ffmpeg_path}")
            return False

        ok = messagebox.askyesno(
            "FFmpeg not found",
            "ffmpeg.exe was not found.\n\nDownload FFmpeg now?",
        )
        if not ok:
            return False

        try:
            ensure_ffmpeg_available(self.paths, auto_download=True, progress=self.log)
            self.log("FFmpeg is ready.")
            return True
        except FfmpegDownloadError as exc:
            messagebox.showerror("FFmpeg download failed", str(exc))
            return False
        except Exception as exc:
            messagebox.showerror("FFmpeg download failed", str(exc))
            return False

    def read_settings(self) -> Optional[EncodeSettings]:
        try:
            max_jobs = int(self.max_jobs_var.get())
            height_4k = int(self.height_4k_var.get())
            cq_value = int(self.cq_var.get())
        except ValueError:
            messagebox.showerror("Error", "Max jobs, 4K height, and CQ must be numbers.")
            return None

        if max_jobs < 1:
            messagebox.showerror("Error", "Max jobs must be 1 or more.")
            return None
        if height_4k < 1:
            messagebox.showerror("Error", "4K height must be 1 or more.")
            return None
        if not self.make_4k_var.get() and not self.make_mp4_var.get():
            messagebox.showerror("Error", "Select Make 4K or Make MP4.")
            return None

        return EncodeSettings(
            max_jobs=max_jobs,
            make_4k=self.make_4k_var.get(),
            make_mp4=self.make_mp4_var.get(),
            height_4k=height_4k,
            codec=self.codec_var.get(),
            preset=self.preset_var.get(),
            tune=self.tune_var.get(),
            rate_mode=self.rate_mode_var.get(),
            cq_value=cq_value,
            bitrate=self.bitrate_var.get().strip(),
            maxrate=self.maxrate_var.get().strip(),
            bufsize=self.bufsize_var.get().strip(),
        )

    def start_selected(self) -> None:
        if not self.ensure_ffmpeg_before_run():
            return

        settings = self.read_settings()
        if settings is None:
            return

        selected = list(self.file_list.curselection())
        if not selected:
            messagebox.showwarning("Warning", "Select one or more files.")
            return

        selected_files = [self.files[i].path for i in selected]
        specs = build_job_specs(self.paths, selected_files, settings)
        self.start_specs(settings, specs, save=True)

    def resume_saved(self) -> None:
        if not self.ensure_ffmpeg_before_run():
            return

        data = load_state(self.paths)
        if not data:
            messagebox.showinfo("Info", "No saved state was found.")
            return

        settings = EncodeSettings.from_dict(data.get("settings", {}))
        specs = resumable_specs(self.paths, data)
        if not specs:
            clear_state(self.paths)
            self.log("Saved jobs are already completed. State was cleared.")
            self.scan_files()
            return

        self.start_specs(settings, specs, save=True)

    def start_specs(self, settings: EncodeSettings, specs: List[JobSpec], save: bool) -> None:
        with self.lock:
            if self.running:
                messagebox.showwarning("Warning", "Jobs are already running.")
                return
            self.running = True
            self.paused = False
            self.stop_requested = False
            self.active_jobs.clear()
            self.all_jobs.clear()
            while not self.pending_jobs.empty():
                try:
                    self.pending_jobs.get_nowait()
                except queue.Empty:
                    break

        if not specs:
            with self.lock:
                self.running = False
            self.log("No jobs were added.")
            self.move_finished_sources(settings)
            self.scan_files()
            return

        if save:
            save_state(self.paths, settings, specs)

        for spec in specs:
            job = self.create_runtime_job(spec, settings)
            self.pending_jobs.put(job)
            self.all_jobs[job.job_id] = job

        self.log(f"Added {len(specs)} job(s).")
        self.scheduler_thread = threading.Thread(target=self.scheduler_loop, args=(settings,), daemon=True)
        self.scheduler_thread.start()

    def create_runtime_job(self, spec: JobSpec, settings: EncodeSettings) -> RuntimeJob:
        self.job_counter += 1
        src = Path(spec.src)
        tmp_out = temp_path_for(self.paths, src, spec.mode)
        out_file = output_path_for(self.paths, src, spec.mode)
        log_file = self.paths.log_dir / f"job_{self.job_counter}_{spec.mode}.log"
        command = build_ffmpeg_command(self.paths.ffmpeg_path, src, tmp_out, spec.mode, settings)
        return RuntimeJob(
            job_id=self.job_counter,
            spec=spec,
            tmp_out=tmp_out,
            out_file=out_file,
            log_file=log_file,
            command=command,
        )

    def scheduler_loop(self, settings: EncodeSettings) -> None:
        self.log("Scheduler started.")

        while True:
            with self.lock:
                should_stop = self.stop_requested
                is_paused = self.paused
                active_count = len(self.active_jobs)
                pending_count = self.pending_jobs.qsize()

            if should_stop:
                self.log("Stop requested.")
                break

            if not is_paused:
                while active_count < settings.max_jobs:
                    try:
                        job = self.pending_jobs.get_nowait()
                    except queue.Empty:
                        break
                    self.start_job_thread(job)
                    with self.lock:
                        active_count = len(self.active_jobs)

            with self.lock:
                active_count = len(self.active_jobs)
                pending_count = self.pending_jobs.qsize()

            if active_count == 0 and pending_count == 0:
                break

            time.sleep(0.3)

        while True:
            with self.lock:
                active_count = len(self.active_jobs)
            if active_count == 0:
                break
            time.sleep(0.3)

        with self.lock:
            stopped = self.stop_requested
            self.running = False
            self.paused = False

        if stopped:
            self.log("Scheduler stopped.")
        else:
            self.log("Scheduler finished.")
            self.move_finished_sources(settings)
            clear_state(self.paths)
            self.root.after(0, self.scan_files)

    def start_job_thread(self, job: RuntimeJob) -> None:
        with self.lock:
            self.active_jobs[job.job_id] = job
            job.status = "running"
        thread = threading.Thread(target=self.run_job, args=(job,), daemon=True)
        thread.start()

    def run_job(self, job: RuntimeJob) -> None:
        src = Path(job.spec.src)
        self.log(f"Start {job.spec.mode}: {src.name}")
        self.log(f"Log file: {job.log_file}")

        if job.tmp_out.exists():
            try:
                job.tmp_out.unlink()
            except Exception as exc:
                self.log(f"Failed to delete temp file: {exc}")

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        try:
            with open(job.log_file, "w", encoding="utf-8", errors="replace") as log_fp:
                log_fp.write("Command:\n")
                log_fp.write(command_to_text(job.command))
                log_fp.write("\n\n")
                log_fp.flush()

                process = subprocess.Popen(
                    job.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
                job.process = process

                if process.stdout is not None:
                    for line in process.stdout:
                        line = line.rstrip("\r\n")
                        if line:
                            log_fp.write(line + "\n")
                            log_fp.flush()
                            self.log(f"job {job.job_id}: {line}")

                ret = process.wait()

            if ret == 0 and job.tmp_out.exists():
                job.out_file.parent.mkdir(parents=True, exist_ok=True)
                if job.out_file.exists():
                    job.out_file.unlink()
                shutil.move(str(job.tmp_out), str(job.out_file))
                job.status = "ok"
                self.log(f"Finish {job.spec.mode}: {src.name}")
            else:
                job.status = "fail"
                if job.tmp_out.exists():
                    job.tmp_out.unlink(missing_ok=True)
                self.log(f"Failed {job.spec.mode}: {src.name} / exit code {ret}")
        except Exception as exc:
            job.status = "fail"
            if job.tmp_out.exists():
                job.tmp_out.unlink(missing_ok=True)
            self.log(f"Exception in job {job.job_id}: {exc}")
        finally:
            with self.lock:
                self.active_jobs.pop(job.job_id, None)

    def pause_jobs(self) -> None:
        with self.lock:
            if not self.running:
                return
            self.paused = True
            jobs = list(self.active_jobs.values())

        if psutil is None:
            self.log("Soft pause only. Install psutil for real process pause.")
            self.log("Soft pause: new jobs will not start.")
            return

        for job in jobs:
            self.suspend_process(job)
        self.log("Paused.")

    def resume_jobs(self) -> None:
        with self.lock:
            if not self.running:
                return
            jobs = list(self.active_jobs.values())
            self.paused = False

        if psutil is not None:
            for job in jobs:
                self.resume_process(job)
        self.log("Resumed.")

    def stop_all(self) -> None:
        with self.lock:
            if not self.running:
                return
            self.stop_requested = True
            self.paused = False
            jobs = list(self.active_jobs.values())
            while not self.pending_jobs.empty():
                try:
                    dropped = self.pending_jobs.get_nowait()
                    dropped.status = "cancelled"
                except queue.Empty:
                    break

        for job in jobs:
            if psutil is not None:
                self.resume_process(job)
                self.kill_process(job)
            elif job.process is not None:
                try:
                    job.process.kill()
                except Exception:
                    pass
        self.log("Stop sent to all active jobs.")

    def suspend_process(self, job: RuntimeJob) -> None:
        if job.process is None or job.process.poll() is not None:
            return
        try:
            parent = psutil.Process(job.process.pid)  # type: ignore[union-attr]
            for child in parent.children(recursive=True):
                try:
                    child.suspend()
                except Exception:
                    pass
            parent.suspend()
        except Exception as exc:
            self.log(f"Pause failed for job {job.job_id}: {exc}")

    def resume_process(self, job: RuntimeJob) -> None:
        if job.process is None:
            return
        try:
            parent = psutil.Process(job.process.pid)  # type: ignore[union-attr]
            for child in parent.children(recursive=True):
                try:
                    child.resume()
                except Exception:
                    pass
            parent.resume()
        except Exception:
            pass

    def kill_process(self, job: RuntimeJob) -> None:
        if job.process is None or job.process.poll() is not None:
            return
        try:
            parent = psutil.Process(job.process.pid)  # type: ignore[union-attr]
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            parent.kill()
        except Exception as exc:
            self.log(f"Kill failed for job {job.job_id}: {exc}")

    def move_finished_sources(self, settings: EncodeSettings) -> None:
        moved = 0
        kept = 0
        for status in scan_obs_files(self.paths):
            src = status.path
            stem = src.stem
            k4_ok = (self.paths.k4_dir / f"{stem}.mp4").exists()
            mp4_ok = (self.paths.mp4_dir / f"{stem}.mp4").exists()

            if settings.make_4k and not k4_ok:
                kept += 1
                self.log(f"Keep source: {src.name} / Missing 4K output.")
                continue
            if settings.make_mp4 and not mp4_ok:
                kept += 1
                self.log(f"Keep source: {src.name} / Missing MP4 output.")
                continue

            try:
                dest = self.paths.source_dir / src.name
                if dest.exists():
                    base = dest.stem
                    suffix = dest.suffix
                    n = 1
                    while dest.exists():
                        dest = self.paths.source_dir / f"{base}_{n}{suffix}"
                        n += 1
                shutil.move(str(src), str(dest))
                moved += 1
                self.log(f"Move source: {src.name}")
            except Exception as exc:
                kept += 1
                self.log(f"Move failed: {src.name} / {exc}")

        self.log(f"Move result: moved={moved}, kept={kept}")


def main() -> None:
    root = tk.Tk()
    app = EncoderApp(root)

    def on_close() -> None:
        if app.running:
            ok = messagebox.askyesno("Exit", "Jobs are running. Stop all jobs and exit?")
            if not ok:
                return
            app.stop_all()
            time.sleep(0.5)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
