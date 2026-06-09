from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from ffmpeg_nvenc_gui.core import (
        AppPaths,
        EncodeProfile,
        FileStatus,
        GpuInfo,
        JobSpec,
        OutputVariant,
        RESOLUTION_PRESETS,
        build_concat_command,
        build_ffmpeg_command,
        build_job_specs,
        build_paths,
        clear_state,
        command_to_text,
        concat_list_path_for,
        default_profile,
        detect_nvidia_gpus,
        duplicate_output_targets,
        duplicate_output_targets_for_outputs,
        ensure_dirs,
        ensure_profile_dirs,
        load_profiles,
        load_state,
        missing_profile_dirs,
        missing_rate_fields,
        new_id,
        normalize_container_extension,
        output_path_for,
        parse_ffmpeg_time,
        partial_segment_path_for,
        probe_duration,
        profile_archive_dir,
        profile_from_state,
        profile_input_dir,
        resumable_specs,
        safe_folder_name,
        save_profiles,
        save_state,
        scan_profile_files,
        segment_dir_for,
        segment_path_for,
        segment_ranges,
        temp_output_path_for,
        variant_by_id,
        write_concat_file,
    )
    from ffmpeg_nvenc_gui.ffmpeg_downloader import FfmpegDownloadError, ensure_ffmpeg_available
except ModuleNotFoundError:
    from core import (  # type: ignore
        AppPaths,
        EncodeProfile,
        FileStatus,
        GpuInfo,
        JobSpec,
        OutputVariant,
        RESOLUTION_PRESETS,
        build_concat_command,
        build_ffmpeg_command,
        build_job_specs,
        build_paths,
        clear_state,
        command_to_text,
        concat_list_path_for,
        default_profile,
        detect_nvidia_gpus,
        duplicate_output_targets,
        duplicate_output_targets_for_outputs,
        ensure_dirs,
        ensure_profile_dirs,
        load_profiles,
        load_state,
        missing_profile_dirs,
        missing_rate_fields,
        new_id,
        normalize_container_extension,
        output_path_for,
        parse_ffmpeg_time,
        partial_segment_path_for,
        probe_duration,
        profile_archive_dir,
        profile_from_state,
        profile_input_dir,
        resumable_specs,
        safe_folder_name,
        save_profiles,
        save_state,
        scan_profile_files,
        segment_dir_for,
        segment_path_for,
        segment_ranges,
        temp_output_path_for,
        variant_by_id,
        write_concat_file,
    )
    from ffmpeg_downloader import FfmpegDownloadError, ensure_ffmpeg_available  # type: ignore


PROFILE_DIR_LABELS = {
    "input_dir": "入力先",
    "output_dir": "出力先",
    "archive_dir": "処理済み退避先",
}


def profile_dir_label_text(fields: List[str]) -> str:
    return ", ".join(PROFILE_DIR_LABELS.get(field, field) for field in fields)


@dataclass
class RuntimeJob:
    job_id: int
    spec: JobSpec
    profile: EncodeProfile
    variant: OutputVariant
    tmp_out: Path
    out_file: Path
    log_file: Path
    process: Optional[subprocess.Popen] = None
    status: str = "waiting"
    message: str = ""
    total_segments: int = 1
    completed_segments: int = 0
    current_segment: int = 0
    progress: float = 0.0


class EncoderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NVEnc Archive Studio")
        self.root.geometry("1220x820")
        self.root.minsize(1060, 700)

        self.paths: AppPaths = build_paths()
        ensure_dirs(self.paths)

        self.gpus: List[GpuInfo] = detect_nvidia_gpus()
        self.profiles: List[EncodeProfile] = load_profiles(self.paths, self.gpus)
        self.active_profile_id: Optional[str] = self.profiles[0].id if self.profiles else None
        self.files: List[FileStatus] = []
        self.editing_outputs: List[OutputVariant] = []
        self.selected_output_id: Optional[str] = None

        self.job_counter = 0
        self.pending_jobs: queue.Queue[RuntimeJob] = queue.Queue()
        self.active_jobs: Dict[int, RuntimeJob] = {}
        self.all_jobs: Dict[int, RuntimeJob] = {}
        self.job_rows: Dict[int, str] = {}

        self.lock = threading.Lock()
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.running = False
        self.paused = False
        self.stop_requested = False
        self.scheduler_thread: Optional[threading.Thread] = None

        self._build_ui()
        self.refresh_profile_choices()
        self.load_profile_into_form(self.current_profile())
        self._poll_log_queue()
        self.scan_files()
        self._announce_resume_state()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.colors = {
            "bg": "#f4f6f8",
            "surface": "#ffffff",
            "text": "#1f2937",
            "muted": "#667085",
            "line": "#d7dde5",
            "accent": "#2563eb",
            "danger": "#b42318",
            "ok": "#067647",
        }

        self.root.configure(bg=self.colors["bg"])
        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure("Surface.TFrame", background=self.colors["surface"], relief="flat")
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])
        style.configure("Surface.TLabel", background=self.colors["surface"], foreground=self.colors["text"])
        style.configure("Muted.TLabel", background=self.colors["surface"], foreground=self.colors["muted"])
        style.configure("Title.TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=self.colors["bg"], foreground=self.colors["muted"], font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=self.colors["surface"], foreground=self.colors["text"], font=("Segoe UI", 11, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Danger.TButton", foreground=self.colors["danger"], font=("Segoe UI", 10, "bold"))
        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(18, 8), font=("Segoe UI", 10))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("Horizontal.TProgressbar", thickness=12)

    def _build_ui(self) -> None:
        self._configure_style()

        self.active_profile_var = tk.StringVar()
        self.status_var = tk.StringVar(value="待機中")
        self.progress_text_var = tk.StringVar(value="0%")
        self.input_summary_var = tk.StringVar(value="")
        self.output_summary_var = tk.StringVar(value="")

        main = ttk.Frame(self.root, padding=(20, 18), style="App.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main, style="App.TFrame")
        header.pack(fill=tk.X)
        title_block = ttk.Frame(header, style="App.TFrame")
        title_block.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title_block, text="NVEnc Archive Studio", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            title_block,
            text="プロファイルで入力先と複数出力を定義し、実行画面は進捗確認に集中します。",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        selector = ttk.Frame(header, style="App.TFrame")
        selector.pack(side=tk.RIGHT)
        ttk.Label(selector, text="実行プロファイル").pack(anchor=tk.W)
        self.profile_combo = ttk.Combobox(
            selector,
            textvariable=self.active_profile_var,
            width=32,
            state="readonly",
        )
        self.profile_combo.pack(anchor=tk.E, pady=(3, 0))
        self.profile_combo.bind("<<ComboboxSelected>>", self.on_profile_selected)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(16, 0))

        self.run_tab = ttk.Frame(self.notebook, padding=14, style="App.TFrame")
        self.profile_tab = ttk.Frame(self.notebook, padding=14, style="App.TFrame")
        self.notebook.add(self.run_tab, text="実行")
        self.notebook.add(self.profile_tab, text="プロファイル")

        self._build_run_tab()
        self._build_profile_tab()

    def _surface(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=14, style="Surface.TFrame")
        return frame

    def _build_run_tab(self) -> None:
        summary = self._surface(self.run_tab)
        summary.pack(fill=tk.X)

        left = ttk.Frame(summary, style="Surface.TFrame")
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, textvariable=self.status_var, style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(left, textvariable=self.input_summary_var, style="Muted.TLabel").pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(left, textvariable=self.output_summary_var, style="Muted.TLabel").pack(anchor=tk.W)

        controls = ttk.Frame(summary, style="Surface.TFrame")
        controls.pack(side=tk.RIGHT)
        ttk.Button(controls, text="開始", style="Accent.TButton", command=self.start_current_profile).pack(side=tk.LEFT, padx=(0, 8))
        self.pause_button = ttk.Button(controls, text="一時停止", command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="中断", style="Danger.TButton", command=self.stop_all).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="保存状態から再開", command=self.resume_saved).pack(side=tk.LEFT)

        progress_box = self._surface(self.run_tab)
        progress_box.pack(fill=tk.X, pady=(12, 0))
        progress_head = ttk.Frame(progress_box, style="Surface.TFrame")
        progress_head.pack(fill=tk.X)
        ttk.Label(progress_head, text="全体進捗", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Label(progress_head, textvariable=self.progress_text_var, style="Muted.TLabel").pack(side=tk.RIGHT)
        self.overall_progress = ttk.Progressbar(progress_box, mode="determinate", maximum=100)
        self.overall_progress.pack(fill=tk.X, pady=(10, 0))

        body = ttk.Frame(self.run_tab, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        list_box = self._surface(body)
        list_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_head = ttk.Frame(list_box, style="Surface.TFrame")
        list_head.pack(fill=tk.X)
        ttk.Label(list_head, text="ファイル別進捗", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(list_head, text="更新", command=self.scan_files).pack(side=tk.RIGHT)

        columns = ("file", "output", "status", "progress")
        self.progress_tree = ttk.Treeview(list_box, columns=columns, show="headings", height=15)
        for key, text, width in [
            ("file", "ファイル", 260),
            ("output", "出力", 170),
            ("status", "状態", 120),
            ("progress", "進捗", 90),
        ]:
            self.progress_tree.heading(key, text=text)
            self.progress_tree.column(key, width=width, anchor=tk.W)
        self.progress_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(10, 0))
        tree_scroll = ttk.Scrollbar(list_box, orient=tk.VERTICAL, command=self.progress_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(10, 0))
        self.progress_tree.config(yscrollcommand=tree_scroll.set)

        log_box = self._surface(body)
        log_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(12, 0))
        log_head = ttk.Frame(log_box, style="Surface.TFrame")
        log_head.pack(fill=tk.X)
        ttk.Label(log_head, text="ログ", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(log_head, text="ログフォルダ", command=self.open_log_dir).pack(side=tk.RIGHT)
        self.log_text = ScrolledText(
            log_box,
            wrap=tk.WORD,
            height=18,
            bg="#0f172a",
            fg="#dbeafe",
            insertbackground="#dbeafe",
            relief=tk.FLAT,
            padx=10,
            pady=10,
            font=("Consolas", 9),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _build_profile_tab(self) -> None:
        form = self._surface(self.profile_tab)
        form.pack(fill=tk.X)

        toolbar = ttk.Frame(form, style="Surface.TFrame")
        toolbar.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, 12))
        ttk.Label(toolbar, text="プロファイル設定", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="新規", command=self.new_profile).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(toolbar, text="削除", command=self.delete_current_profile).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(toolbar, text="保存", style="Accent.TButton", command=self.save_current_profile).pack(side=tk.RIGHT)

        self.profile_name_var = tk.StringVar()
        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.archive_dir_var = tk.StringVar()
        self.max_jobs_var = tk.IntVar(value=2)
        self.segment_minutes_var = tk.IntVar(value=10)
        self.gpu_choice_var = tk.StringVar(value="CPU only")
        self.codec_var = tk.StringVar(value="hevc_nvenc")
        self.cpu_codec_var = tk.StringVar(value="libx264")
        self.preset_var = tk.StringVar(value="p7")
        self.tune_var = tk.StringVar(value="hq")
        self.rate_mode_var = tk.StringVar(value="CQ")
        self.cq_var = tk.StringVar(value="18")
        self.bitrate_var = tk.StringVar(value="25000k")
        self.maxrate_var = tk.StringVar(value="40000k")
        self.bufsize_var = tk.StringVar(value="80000k")

        self._label_entry(form, "名前", self.profile_name_var, 1, 0, width=28)
        self._path_entry(form, "入力先", self.input_dir_var, 2, 0)
        self._path_entry(form, "出力先", self.output_dir_var, 3, 0)
        self._path_entry(form, "処理済み退避先", self.archive_dir_var, 4, 0)

        ttk.Label(form, text="同時実行数", style="Surface.TLabel").grid(row=1, column=3, sticky=tk.W, padx=(24, 4))
        ttk.Spinbox(form, from_=1, to=8, textvariable=self.max_jobs_var, width=6).grid(row=1, column=4, sticky=tk.W)
        ttk.Label(form, text="分割間隔(分)", style="Surface.TLabel").grid(row=2, column=3, sticky=tk.W, padx=(24, 4))
        ttk.Spinbox(form, from_=1, to=120, textvariable=self.segment_minutes_var, width=6).grid(row=2, column=4, sticky=tk.W)

        ttk.Label(form, text="GPU", style="Surface.TLabel").grid(row=3, column=3, sticky=tk.W, padx=(24, 4))
        self.gpu_combo = ttk.Combobox(form, textvariable=self.gpu_choice_var, state="readonly", width=34)
        self.gpu_combo.grid(row=3, column=4, columnspan=2, sticky="ew")

        ttk.Label(form, text="NVENC Codec", style="Surface.TLabel").grid(row=5, column=0, sticky=tk.W, pady=(12, 0))
        ttk.Combobox(
            form,
            textvariable=self.codec_var,
            values=["hevc_nvenc", "h264_nvenc", "av1_nvenc"],
            width=18,
            state="readonly",
        ).grid(row=5, column=1, sticky=tk.W, pady=(12, 0))
        ttk.Label(form, text="CPU Codec", style="Surface.TLabel").grid(row=5, column=2, sticky=tk.W, pady=(12, 0), padx=(16, 4))
        ttk.Combobox(
            form,
            textvariable=self.cpu_codec_var,
            values=["libx264", "libx265"],
            width=14,
            state="readonly",
        ).grid(row=5, column=3, sticky=tk.W, pady=(12, 0))

        ttk.Label(form, text="Preset", style="Surface.TLabel").grid(row=6, column=0, sticky=tk.W)
        ttk.Combobox(
            form,
            textvariable=self.preset_var,
            values=["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
            width=8,
            state="readonly",
        ).grid(row=6, column=1, sticky=tk.W)
        ttk.Label(form, text="Tune", style="Surface.TLabel").grid(row=6, column=2, sticky=tk.W, padx=(16, 4))
        ttk.Combobox(
            form,
            textvariable=self.tune_var,
            values=["none", "hq", "ll", "ull", "lossless"],
            width=12,
            state="readonly",
        ).grid(row=6, column=3, sticky=tk.W)
        ttk.Label(form, text="Rate", style="Surface.TLabel").grid(row=6, column=4, sticky=tk.W, padx=(16, 4))
        rate_combo = ttk.Combobox(
            form,
            textvariable=self.rate_mode_var,
            values=["CQ", "VBR", "ABR", "CBR"],
            width=8,
            state="readonly",
        )
        rate_combo.grid(row=6, column=5, sticky=tk.W)
        rate_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_rate_controls())

        self.cq_entry = self._compact_entry(form, "CQ/CRF", self.cq_var, 7, 0)
        self.bitrate_entry = self._compact_entry(form, "Bitrate", self.bitrate_var, 7, 2)
        self.maxrate_entry = self._compact_entry(form, "Maxrate", self.maxrate_var, 7, 4)
        self.bufsize_entry = self._compact_entry(form, "Bufsize", self.bufsize_var, 8, 0)

        for column in range(6):
            form.columnconfigure(column, weight=1)

        outputs = self._surface(self.profile_tab)
        outputs.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        output_head = ttk.Frame(outputs, style="Surface.TFrame")
        output_head.pack(fill=tk.X)
        ttk.Label(output_head, text="出力バリアント", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(output_head, text="選択を解除", command=self.clear_output_selection).pack(side=tk.RIGHT)

        output_body = ttk.Frame(outputs, style="Surface.TFrame")
        output_body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        columns = ("name", "resolution", "folder", "container")
        self.outputs_tree = ttk.Treeview(output_body, columns=columns, show="headings", height=8)
        for key, text, width in [
            ("name", "名前", 190),
            ("resolution", "解像度", 100),
            ("folder", "フォルダ名", 180),
            ("container", "形式", 70),
        ]:
            self.outputs_tree.heading(key, text=text)
            self.outputs_tree.column(key, width=width, anchor=tk.W)
        self.outputs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.outputs_tree.bind("<<TreeviewSelect>>", self.on_output_select)

        edit = ttk.Frame(output_body, padding=(16, 0, 0, 0), style="Surface.TFrame")
        edit.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_name_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.output_resolution_var = tk.StringVar(value="1080p")
        self.output_custom_height_var = tk.StringVar(value="")
        self.output_container_var = tk.StringVar(value="mp4")

        self._stacked_label_entry(edit, "名前", self.output_name_var)
        self._stacked_label_entry(edit, "フォルダ名", self.output_folder_var)
        ttk.Label(edit, text="解像度", style="Surface.TLabel").pack(anchor=tk.W, pady=(8, 2))
        resolution_combo = ttk.Combobox(
            edit,
            textvariable=self.output_resolution_var,
            values=list(RESOLUTION_PRESETS.keys()),
            width=22,
            state="readonly",
        )
        resolution_combo.pack(anchor=tk.W)
        resolution_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_resolution_controls())
        ttk.Label(edit, text="カスタム高さ", style="Surface.TLabel").pack(anchor=tk.W, pady=(8, 2))
        self.custom_height_entry = ttk.Entry(edit, textvariable=self.output_custom_height_var, width=24)
        self.custom_height_entry.pack(anchor=tk.W)
        self._stacked_label_entry(edit, "形式", self.output_container_var)
        ttk.Button(edit, text="追加/更新", style="Accent.TButton", command=self.add_or_update_output).pack(fill=tk.X, pady=(12, 4))
        ttk.Button(edit, text="削除", command=self.remove_output).pack(fill=tk.X)

        footer = ttk.Frame(self.profile_tab, style="App.TFrame")
        footer.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(footer, text="FFmpegを確認/導入", command=self.download_ffmpeg_button).pack(side=tk.RIGHT)

        self.update_rate_controls()
        self.update_resolution_controls()

    def _label_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, column: int, width: int = 38) -> ttk.Entry:
        ttk.Label(parent, text=label, style="Surface.TLabel").grid(row=row, column=column, sticky=tk.W, pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=column + 1, sticky="ew", pady=3)
        return entry

    def _path_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, column: int) -> None:
        ttk.Label(parent, text=label, style="Surface.TLabel").grid(row=row, column=column, sticky=tk.W, pady=3)
        entry = ttk.Entry(parent, textvariable=variable, width=48)
        entry.grid(row=row, column=column + 1, columnspan=2, sticky="ew", pady=3)
        ttk.Button(parent, text="選択", command=lambda: self.browse_dir(variable)).grid(row=row, column=column + 3, sticky=tk.W, padx=(8, 0), pady=3)

    def _compact_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, column: int) -> ttk.Entry:
        ttk.Label(parent, text=label, style="Surface.TLabel").grid(row=row, column=column, sticky=tk.W, pady=(8, 0))
        entry = ttk.Entry(parent, textvariable=variable, width=12)
        entry.grid(row=row, column=column + 1, sticky=tk.W, pady=(8, 0))
        return entry

    def _stacked_label_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> ttk.Entry:
        ttk.Label(parent, text=label, style="Surface.TLabel").pack(anchor=tk.W, pady=(0, 2))
        entry = ttk.Entry(parent, textvariable=variable, width=26)
        entry.pack(anchor=tk.W, pady=(0, 8))
        return entry

    def browse_dir(self, variable: tk.StringVar) -> None:
        value = filedialog.askdirectory(initialdir=variable.get() or str(self.paths.base_dir))
        if value:
            variable.set(value)

    def _profile_labels(self) -> List[str]:
        counts: Dict[str, int] = {}
        for profile in self.profiles:
            counts[profile.name] = counts.get(profile.name, 0) + 1

        labels = []
        for profile in self.profiles:
            if counts.get(profile.name, 0) > 1:
                labels.append(f"{profile.name} ({self.profile_short_id(profile)})")
            else:
                labels.append(profile.name)
        return labels

    def profile_short_id(self, profile: EncodeProfile) -> str:
        return profile.id.rsplit("_", 1)[-1][:8]

    def refresh_profile_choices(self) -> None:
        values = self._profile_labels()
        self.profile_combo.configure(values=values)
        self.gpu_combo.configure(values=self._gpu_choices())

        profile_ids = [profile.id for profile in self.profiles]
        if self.active_profile_id not in profile_ids:
            self.active_profile_id = profile_ids[0] if profile_ids else None

        active_index = self.profile_index_by_id(self.active_profile_id)
        if active_index is None:
            self.active_profile_var.set("")
            return

        self.active_profile_var.set(values[active_index])
        self.profile_combo.current(active_index)

    def _gpu_choices(self) -> List[str]:
        choices = ["CPU only"]
        choices.extend([f"GPU {gpu.index}: {gpu.name}" for gpu in self.gpus])
        return choices

    def profile_index_by_id(self, profile_id: Optional[str]) -> Optional[int]:
        if profile_id is None:
            return None
        for index, profile in enumerate(self.profiles):
            if profile.id == profile_id:
                return index
        return None

    def current_profile(self) -> EncodeProfile:
        selected_index = self.profile_index_by_id(self.active_profile_id)
        if selected_index is not None:
            return self.profiles[selected_index]

        combo_index = self.profile_combo.current()
        if 0 <= combo_index < len(self.profiles):
            self.active_profile_id = self.profiles[combo_index].id
            return self.profiles[combo_index]

        self.active_profile_id = self.profiles[0].id
        return self.profiles[0]

    def unique_profile_name(self, prefix: str = "Profile") -> str:
        existing = {profile.name for profile in self.profiles}
        index = len(self.profiles) + 1
        while True:
            candidate = f"{prefix} {index}"
            if candidate not in existing:
                return candidate
            index += 1

    def has_duplicate_profile_name(self, profile_id: str, name: str) -> bool:
        for profile in self.profiles:
            if profile.id != profile_id and profile.name == name:
                return True
        return False

    def on_profile_selected(self, _event: object = None) -> None:
        combo_index = self.profile_combo.current()
        if 0 <= combo_index < len(self.profiles):
            self.active_profile_id = self.profiles[combo_index].id
        if self.running:
            self.log("実行中のため、表示プロファイルだけ切り替えます。")
        self.load_profile_into_form(self.current_profile())
        self.scan_files()

    def load_profile_into_form(self, profile: EncodeProfile) -> None:
        self.profile_name_var.set(profile.name)
        self.input_dir_var.set(profile.input_dir)
        self.output_dir_var.set(profile.output_dir)
        self.archive_dir_var.set(profile.archive_dir)
        self.max_jobs_var.set(profile.max_parallel_jobs)
        self.segment_minutes_var.set(profile.segment_minutes)
        self.gpu_choice_var.set(self._choice_for_profile_gpu(profile))
        self.codec_var.set(profile.codec)
        self.cpu_codec_var.set(profile.cpu_codec)
        self.preset_var.set(profile.preset)
        self.tune_var.set(profile.tune)
        self.rate_mode_var.set(profile.rate_mode)
        self.cq_var.set(str(profile.cq_value))
        self.bitrate_var.set(profile.bitrate)
        self.maxrate_var.set(profile.maxrate)
        self.bufsize_var.set(profile.bufsize)
        self.editing_outputs = [OutputVariant.from_dict(asdict(item)) for item in profile.outputs]
        self.selected_output_id = None
        self.refresh_outputs_tree()
        self.update_rate_controls()

    def _choice_for_profile_gpu(self, profile: EncodeProfile) -> str:
        if not profile.use_gpu:
            return "CPU only"
        for gpu in self.gpus:
            if gpu.index == profile.gpu_index:
                return f"GPU {gpu.index}: {gpu.name}"
        return "CPU only"

    def _parse_gpu_choice(self) -> tuple[bool, int, str]:
        choice = self.gpu_choice_var.get()
        if not choice.startswith("GPU "):
            return False, 0, ""
        prefix, _, name = choice.partition(":")
        index_text = prefix.replace("GPU", "").strip()
        if not index_text.isdigit():
            return False, 0, ""
        return True, int(index_text), name.strip()

    def update_rate_controls(self) -> None:
        mode = self.rate_mode_var.get().upper()
        controls = {
            self.cq_entry: mode in {"CQ", "VBR"},
            self.bitrate_entry: mode in {"VBR", "ABR", "CBR"},
            self.maxrate_entry: mode == "VBR",
            self.bufsize_entry: mode in {"VBR", "CBR"},
        }
        for widget, enabled in controls.items():
            widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def update_resolution_controls(self) -> None:
        state = tk.NORMAL if self.output_resolution_var.get() == "Custom" else tk.DISABLED
        self.custom_height_entry.configure(state=state)

    def collect_profile_from_form(self) -> Optional[EncodeProfile]:
        current = self.current_profile()
        mode = self.rate_mode_var.get().upper()
        try:
            max_jobs = int(self.max_jobs_var.get())
            segment_minutes = int(self.segment_minutes_var.get())
        except ValueError:
            messagebox.showerror("入力エラー", "同時実行数と分割間隔は数値で入力してください。")
            return None

        if mode in {"CQ", "VBR"}:
            try:
                cq_value = int(self.cq_var.get())
            except ValueError:
                messagebox.showerror("入力エラー", "CQ/CRF は数値で入力してください。")
                return None
        else:
            cq_value = current.cq_value

        if max_jobs < 1:
            messagebox.showerror("入力エラー", "同時実行数は1以上にしてください。")
            return None
        if segment_minutes < 1:
            messagebox.showerror("入力エラー", "分割間隔は1分以上にしてください。")
            return None
        if not self.editing_outputs:
            messagebox.showerror("入力エラー", "出力バリアントを1つ以上登録してください。")
            return None

        input_dir = self.input_dir_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        archive_dir = self.archive_dir_var.get().strip()
        if not input_dir or not output_dir or not archive_dir:
            messagebox.showerror("入力エラー", "入力先、出力先、処理済み退避先を入力してください。")
            return None

        use_gpu, gpu_index, gpu_name = self._parse_gpu_choice()
        profile = EncodeProfile(
            id=current.id,
            name=self.profile_name_var.get().strip() or current.name,
            input_dir=input_dir,
            output_dir=output_dir,
            archive_dir=archive_dir,
            max_parallel_jobs=max_jobs,
            segment_minutes=segment_minutes,
            use_gpu=use_gpu,
            gpu_index=gpu_index,
            gpu_name=gpu_name,
            codec=self.codec_var.get(),
            cpu_codec=self.cpu_codec_var.get(),
            preset=self.preset_var.get(),
            tune=self.tune_var.get(),
            rate_mode=self.rate_mode_var.get(),
            cq_value=cq_value,
            bitrate=self.bitrate_var.get().strip(),
            maxrate=self.maxrate_var.get().strip(),
            bufsize=self.bufsize_var.get().strip(),
            outputs=[OutputVariant.from_dict(asdict(item)) for item in self.editing_outputs],
        )
        missing = missing_rate_fields(profile)
        if missing:
            labels = ", ".join(missing)
            messagebox.showerror("入力エラー", f"{profile.rate_mode} では {labels} を入力してください。")
            return None
        duplicates = duplicate_output_targets(profile)
        if duplicates:
            messagebox.showerror("入力エラー", f"同じ出力先が重複しています: {', '.join(duplicates)}")
            return None
        return profile

    def save_current_profile(self) -> None:
        profile = self.collect_profile_from_form()
        if profile is None:
            return
        if self.has_duplicate_profile_name(profile.id, profile.name):
            messagebox.showerror("入力エラー", "同じ名前のプロファイルがすでにあります。別の名前にしてください。")
            return

        for index, existing in enumerate(self.profiles):
            if existing.id == profile.id:
                self.profiles[index] = profile
                break
        else:
            self.profiles.append(profile)

        save_profiles(self.paths, self.profiles)
        self.active_profile_id = profile.id
        self.refresh_profile_choices()
        self.log(f"プロファイルを保存: {profile.name}")
        self.scan_files()

    def new_profile(self) -> None:
        profile = default_profile(self.paths, self.gpus)
        profile.id = new_id("profile")
        profile.name = self.unique_profile_name("Profile")
        self.profiles.append(profile)
        save_profiles(self.paths, self.profiles)
        self.active_profile_id = profile.id
        self.refresh_profile_choices()
        self.load_profile_into_form(profile)
        self.scan_files()

    def delete_current_profile(self) -> None:
        if len(self.profiles) <= 1:
            messagebox.showwarning("削除できません", "プロファイルは最低1つ必要です。")
            return
        profile = self.current_profile()
        ok = messagebox.askyesno("削除", f"プロファイル '{profile.name}' を削除しますか？")
        if not ok:
            return
        self.profiles = [item for item in self.profiles if item.id != profile.id]
        save_profiles(self.paths, self.profiles)
        self.active_profile_id = self.profiles[0].id
        self.refresh_profile_choices()
        self.load_profile_into_form(self.current_profile())
        self.scan_files()

    def refresh_outputs_tree(self) -> None:
        self.outputs_tree.delete(*self.outputs_tree.get_children())
        for variant in self.editing_outputs:
            resolution = "Original" if variant.height is None else f"{variant.height}p"
            self.outputs_tree.insert(
                "",
                tk.END,
                iid=variant.id,
                values=(variant.name, resolution, variant.folder_name, variant.container),
            )

    def on_output_select(self, _event: object = None) -> None:
        selection = self.outputs_tree.selection()
        if not selection:
            return
        self.selected_output_id = selection[0]
        for variant in self.editing_outputs:
            if variant.id == self.selected_output_id:
                self.output_name_var.set(variant.name)
                self.output_folder_var.set(variant.folder_name)
                if variant.height is None:
                    self.output_resolution_var.set("Original")
                    self.output_custom_height_var.set("")
                elif variant.height in [2160, 1440, 1080, 720]:
                    self.output_resolution_var.set(f"{variant.height}p")
                    self.output_custom_height_var.set("")
                else:
                    self.output_resolution_var.set("Custom")
                    self.output_custom_height_var.set(str(variant.height))
                self.output_container_var.set(variant.container)
                self.update_resolution_controls()
                return

    def clear_output_selection(self) -> None:
        self.outputs_tree.selection_remove(self.outputs_tree.selection())
        self.selected_output_id = None
        self.output_name_var.set("")
        self.output_folder_var.set("")
        self.output_resolution_var.set("1080p")
        self.output_custom_height_var.set("")
        self.output_container_var.set("mp4")
        self.update_resolution_controls()

    def add_or_update_output(self) -> None:
        name = self.output_name_var.get().strip()
        folder_name = safe_folder_name(self.output_folder_var.get().strip() or name)
        raw_container = self.output_container_var.get()
        container = normalize_container_extension(raw_container, default="")
        if not name:
            messagebox.showerror("入力エラー", "出力名を入力してください。")
            return
        if not container:
            messagebox.showerror("入力エラー", "形式は英数字1〜8文字で入力してください。例: mp4, mkv")
            return

        preset = self.output_resolution_var.get()
        if preset == "Custom":
            try:
                height = int(self.output_custom_height_var.get())
            except ValueError:
                messagebox.showerror("入力エラー", "カスタム解像度の高さを数値で入力してください。")
                return
            if height < 1:
                messagebox.showerror("入力エラー", "カスタム解像度は1以上にしてください。")
                return
        else:
            height = RESOLUTION_PRESETS.get(preset)

        variant = OutputVariant(
            id=self.selected_output_id or new_id("variant"),
            name=name,
            folder_name=folder_name,
            height=height,
            container=container,
            enabled=True,
        )

        next_outputs: List[OutputVariant] = []
        replaced = False
        for existing in self.editing_outputs:
            if existing.id == variant.id:
                next_outputs.append(variant)
                replaced = True
            else:
                next_outputs.append(existing)
        if not replaced:
            next_outputs.append(variant)

        duplicates = duplicate_output_targets_for_outputs(next_outputs)
        if duplicates:
            messagebox.showerror("入力エラー", f"同じ出力先が重複しています: {', '.join(duplicates)}")
            return

        self.editing_outputs = next_outputs

        self.selected_output_id = variant.id
        self.refresh_outputs_tree()
        self.outputs_tree.selection_set(variant.id)

    def remove_output(self) -> None:
        if not self.selected_output_id:
            return
        self.editing_outputs = [item for item in self.editing_outputs if item.id != self.selected_output_id]
        self.clear_output_selection()
        self.refresh_outputs_tree()

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
            state = "一時停止" if self.paused else ("実行中" if self.running else "待機中")
            jobs = list(self.all_jobs.values())

        self.status_var.set(f"{state} / 実行中 {active} / 待機 {pending}")
        self.pause_button.configure(text="再開" if self.paused else "一時停止")
        self._update_runtime_rows(jobs)
        self._update_overall_progress(jobs)
        self.root.after(250, self._poll_log_queue)

    def _update_overall_progress(self, jobs: List[RuntimeJob]) -> None:
        if not jobs:
            self.overall_progress["value"] = 0
            self.progress_text_var.set("0%")
            return
        value = sum(max(0.0, min(100.0, job.progress)) for job in jobs) / len(jobs)
        self.overall_progress["value"] = value
        self.progress_text_var.set(f"{value:.0f}%")

    def _update_runtime_rows(self, jobs: List[RuntimeJob]) -> None:
        for job in jobs:
            row_id = self.job_rows.get(job.job_id)
            if not row_id or not self.progress_tree.exists(row_id):
                continue
            file_name = Path(job.spec.src).name
            detail = job.message or f"{job.completed_segments}/{job.total_segments} segments"
            self.progress_tree.item(
                row_id,
                values=(file_name, job.variant.name, job.status, f"{job.progress:.0f}%  {detail}"),
            )

    def _announce_resume_state(self) -> None:
        data = load_state(self.paths)
        if data:
            self.log("未完了の保存状態があります。必要なら「保存状態から再開」を押してください。")

    def scan_files(self) -> None:
        if self.running:
            return
        profile = self.current_profile()
        missing_dirs = missing_profile_dirs(profile)
        variants = ", ".join(variant.name for variant in profile.outputs if variant.enabled)
        self.input_summary_var.set(f"入力: {profile.input_dir}")
        self.output_summary_var.set(f"出力: {profile.output_dir} / {variants or '未設定'}")
        if missing_dirs:
            self.files = []
            self.render_scan_rows(profile)
            self.log(f"{profile.name}: フォルダ設定が不足しています: {profile_dir_label_text(missing_dirs)}")
            return
        self.files = scan_profile_files(profile)
        self.render_scan_rows(profile)
        if not profile_input_dir(profile).exists():
            self.log(f"入力フォルダはまだありません: {profile.input_dir}")
        else:
            self.log(f"{profile.name}: {len(self.files)} file(s) scanned.")

    def render_scan_rows(self, profile: EncodeProfile) -> None:
        self.progress_tree.delete(*self.progress_tree.get_children())
        for item in self.files:
            for variant in profile.outputs:
                if not variant.enabled:
                    continue
                done = item.outputs.get(variant.id, False)
                self.progress_tree.insert(
                    "",
                    tk.END,
                    values=(
                        item.path.name,
                        variant.name,
                        "完了" if done else "待機",
                        "100%" if done else "0%",
                    ),
                )

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
            self.log("FFmpeg / FFprobe is ready.")
        except Exception as exc:
            self.log(f"FFmpeg install failed: {exc}")
            self.root.after(0, lambda: messagebox.showerror("FFmpeg install failed", str(exc)))

    def ensure_ffmpeg_before_run(self) -> bool:
        if self.paths.ffmpeg_path.exists() and self.paths.ffprobe_path.exists():
            try:
                ensure_ffmpeg_available(self.paths, auto_download=False, progress=self.log)
                return True
            except Exception as exc:
                messagebox.showerror("FFmpeg error", str(exc))
                return False

        messagebox.showinfo(
            "FFmpeg を準備します",
            "FFmpeg / FFprobe が見つからないため、自動でダウンロードして配置します。",
        )
        try:
            ensure_ffmpeg_available(self.paths, auto_download=True, progress=self.log)
            self.log("FFmpeg / FFprobe is ready.")
            return True
        except FfmpegDownloadError as exc:
            messagebox.showerror("FFmpeg install failed", str(exc))
            return False
        except Exception as exc:
            messagebox.showerror("FFmpeg install failed", str(exc))
            return False

    def validate_profile_before_run(self, profile: EncodeProfile) -> bool:
        missing_dirs = missing_profile_dirs(profile)
        if missing_dirs:
            messagebox.showerror("入力エラー", f"{profile_dir_label_text(missing_dirs)} を入力してください。")
            return False
        missing = missing_rate_fields(profile)
        if missing:
            labels = ", ".join(missing)
            messagebox.showerror("入力エラー", f"{profile.rate_mode} では {labels} を入力してください。")
            return False
        duplicates = duplicate_output_targets(profile)
        if duplicates:
            messagebox.showerror("入力エラー", f"同じ出力先が重複しています: {', '.join(duplicates)}")
            return False
        return True

    def start_current_profile(self) -> None:
        profile = self.current_profile()
        if not self.validate_profile_before_run(profile):
            return

        ensure_profile_dirs(profile)
        self.files = scan_profile_files(profile)
        if not self.files:
            messagebox.showwarning("対象なし", "入力フォルダに動画ファイルがありません。")
            self.scan_files()
            return

        specs = build_job_specs(profile, [item.path for item in self.files])
        if not specs:
            messagebox.showinfo("対象なし", "このプロファイルの出力はすべて完了しています。")
            self.scan_files()
            return

        if not self.ensure_ffmpeg_before_run():
            return

        self.start_specs(profile, specs, save=True)

    def resume_saved(self) -> None:
        data = load_state(self.paths)
        if not data:
            messagebox.showinfo("保存状態なし", "再開できる保存状態はありません。")
            return

        profile = profile_from_state(data, self.paths, self.gpus)
        if not self.validate_profile_before_run(profile):
            return
        specs = resumable_specs(profile, data)
        if not specs:
            clear_state(self.paths)
            self.log("保存状態はすでに完了済みでした。状態ファイルを削除しました。")
            self.scan_files()
            return

        if not self.ensure_ffmpeg_before_run():
            return

        ensure_profile_dirs(profile)
        self.start_specs(profile, specs, save=True)

    def start_specs(self, profile: EncodeProfile, specs: List[JobSpec], save: bool) -> None:
        with self.lock:
            if self.running:
                messagebox.showwarning("実行中", "すでにジョブが実行中です。")
                return
            self.running = True
            self.paused = False
            self.stop_requested = False
            self.active_jobs.clear()
            self.all_jobs.clear()
            self.job_rows.clear()
            while not self.pending_jobs.empty():
                try:
                    self.pending_jobs.get_nowait()
                except queue.Empty:
                    break

        if save:
            save_state(self.paths, profile, specs)

        self.progress_tree.delete(*self.progress_tree.get_children())
        for spec in specs:
            job = self.create_runtime_job(spec, profile)
            self.pending_jobs.put(job)
            self.all_jobs[job.job_id] = job
            self.job_rows[job.job_id] = self.progress_tree.insert(
                "",
                tk.END,
                values=(Path(spec.src).name, job.variant.name, "待機", "0%"),
            )

        self.log(f"{profile.name}: {len(specs)} job(s) added.")
        self.scheduler_thread = threading.Thread(target=self.scheduler_loop, args=(profile,), daemon=True)
        self.scheduler_thread.start()

    def create_runtime_job(self, spec: JobSpec, profile: EncodeProfile) -> RuntimeJob:
        self.job_counter += 1
        src = Path(spec.src)
        variant = variant_by_id(profile, spec.variant_id)
        tmp_out = temp_output_path_for(self.paths, src, profile, variant)
        out_file = output_path_for(profile, src, variant)
        log_file = self.paths.log_dir / f"job_{self.job_counter}_{variant.folder_name}.log"
        return RuntimeJob(
            job_id=self.job_counter,
            spec=spec,
            profile=profile,
            variant=variant,
            tmp_out=tmp_out,
            out_file=out_file,
            log_file=log_file,
        )

    def scheduler_loop(self, profile: EncodeProfile) -> None:
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
                while active_count < profile.max_parallel_jobs:
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
            self.log("Scheduler stopped. 完了済みセグメントは再開用に残します。")
        else:
            self.log("Scheduler finished.")
            self.move_finished_sources(profile)
            clear_state(self.paths)
            self.root.after(0, self.scan_files)

    def start_job_thread(self, job: RuntimeJob) -> None:
        with self.lock:
            self.active_jobs[job.job_id] = job
            job.status = "準備中"
        thread = threading.Thread(target=self.run_job, args=(job,), daemon=True)
        thread.start()

    def run_job(self, job: RuntimeJob) -> None:
        src = Path(job.spec.src)
        segment_seconds = max(60, int(job.profile.segment_minutes) * 60)
        self.log(f"Start {job.variant.name}: {src.name}")
        self.log(f"Log file: {job.log_file}")

        segment_dir_for(self.paths, src, job.profile, job.variant).mkdir(parents=True, exist_ok=True)

        try:
            with open(job.log_file, "w", encoding="utf-8", errors="replace") as log_fp:
                duration = probe_duration(self.paths.ffprobe_path, src)
                ranges = segment_ranges(duration, segment_seconds)

                with self.lock:
                    job.total_segments = len(ranges)
                    job.message = f"0/{len(ranges)} segments"
                    job.status = "実行中"

                segment_files: List[Path] = []
                for index, (start, duration_seconds) in enumerate(ranges):
                    final_segment = segment_path_for(self.paths, src, job.profile, job.variant, index)
                    partial_segment = partial_segment_path_for(self.paths, src, job.profile, job.variant, index)
                    segment_files.append(final_segment)

                    if final_segment.exists():
                        self._mark_segment_done(job, index + 1)
                        continue

                    if not self.wait_until_unpaused(job):
                        self._mark_job_cancelled(job)
                        return

                    if partial_segment.exists():
                        partial_segment.unlink(missing_ok=True)

                    command = build_ffmpeg_command(
                        self.paths.ffmpeg_path,
                        src,
                        partial_segment,
                        job.profile,
                        job.variant,
                        start_seconds=start,
                        duration_seconds=duration_seconds,
                    )
                    log_fp.write(f"\nSegment {index + 1}/{len(ranges)} command:\n")
                    log_fp.write(command_to_text(command) + "\n\n")
                    log_fp.flush()

                    with self.lock:
                        job.current_segment = index + 1
                        job.status = "実行中"
                        job.message = f"segment {index + 1}/{len(ranges)}"

                    ret = self.run_process(job, command, log_fp, duration_seconds)
                    if self.was_stopped() or ret != 0 or not partial_segment.exists():
                        partial_segment.unlink(missing_ok=True)
                        if self.was_stopped():
                            self._mark_job_cancelled(job)
                        else:
                            self._mark_job_failed(job, f"segment {index + 1} failed: exit {ret}")
                        return

                    partial_segment.replace(final_segment)
                    self._mark_segment_done(job, index + 1)

                if not self.wait_until_unpaused(job):
                    self._mark_job_cancelled(job)
                    return

                concat_file = concat_list_path_for(self.paths, src, job.profile, job.variant)
                write_concat_file(concat_file, segment_files)
                if job.tmp_out.exists():
                    job.tmp_out.unlink(missing_ok=True)
                concat_command = build_concat_command(self.paths.ffmpeg_path, concat_file, job.tmp_out)
                log_fp.write("\nConcat command:\n")
                log_fp.write(command_to_text(concat_command) + "\n\n")
                log_fp.flush()

                with self.lock:
                    job.status = "結合中"
                    job.message = "finalizing"

                ret = self.run_process(job, concat_command, log_fp, None)
                if self.was_stopped() or ret != 0 or not job.tmp_out.exists():
                    job.tmp_out.unlink(missing_ok=True)
                    if self.was_stopped():
                        self._mark_job_cancelled(job)
                    else:
                        self._mark_job_failed(job, f"concat failed: exit {ret}")
                    return

                job.out_file.parent.mkdir(parents=True, exist_ok=True)
                if job.out_file.exists():
                    job.out_file.unlink()
                shutil.move(str(job.tmp_out), str(job.out_file))
                shutil.rmtree(segment_dir_for(self.paths, src, job.profile, job.variant), ignore_errors=True)
                with self.lock:
                    job.status = "完了"
                    job.progress = 100.0
                    job.message = "done"
                self.log(f"Finish {job.variant.name}: {src.name}")
        except Exception as exc:
            self._mark_job_failed(job, str(exc))
            self.log(f"Exception in job {job.job_id}: {exc}")
        finally:
            with self.lock:
                self.active_jobs.pop(job.job_id, None)

    def run_process(
        self,
        job: RuntimeJob,
        command: List[str],
        log_fp,
        segment_duration: Optional[float],
    ) -> int:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
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
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                if line:
                    log_fp.write(line + "\n")
                    log_fp.flush()
                    self.log(f"job {job.job_id}: {line}")
                    self._update_job_progress_from_line(job, line, segment_duration)

        ret = process.wait()
        job.process = None
        return ret

    def _update_job_progress_from_line(
        self,
        job: RuntimeJob,
        line: str,
        segment_duration: Optional[float],
    ) -> None:
        if not segment_duration:
            return
        current = parse_ffmpeg_time(line)
        if current is None:
            return
        current_ratio = min(max(current / segment_duration, 0.0), 1.0)
        with self.lock:
            job.progress = ((job.completed_segments + current_ratio) / max(job.total_segments, 1)) * 100.0

    def _mark_segment_done(self, job: RuntimeJob, completed: int) -> None:
        with self.lock:
            job.completed_segments = max(job.completed_segments, completed)
            job.progress = (job.completed_segments / max(job.total_segments, 1)) * 100.0
            job.message = f"{job.completed_segments}/{job.total_segments} segments"

    def _mark_job_failed(self, job: RuntimeJob, message: str) -> None:
        with self.lock:
            job.status = "失敗"
            job.message = message
        self.log(f"Failed {job.variant.name}: {Path(job.spec.src).name} / {message}")

    def _mark_job_cancelled(self, job: RuntimeJob) -> None:
        with self.lock:
            job.status = "中断"
            job.message = "resume available"
        self.log(f"Cancelled {job.variant.name}: {Path(job.spec.src).name}")

    def was_stopped(self) -> bool:
        with self.lock:
            return self.stop_requested

    def wait_until_unpaused(self, job: RuntimeJob) -> bool:
        while True:
            with self.lock:
                if self.stop_requested:
                    return False
                paused = self.paused
                if paused:
                    job.status = "一時停止"
                    job.message = "waiting between segments"
            if not paused:
                with self.lock:
                    if job.status == "一時停止":
                        job.status = "実行中"
                return True
            time.sleep(0.3)

    def toggle_pause(self) -> None:
        with self.lock:
            if not self.running:
                return
            self.paused = not self.paused
            paused = self.paused
        if paused:
            self.log("安全な一時停止を予約しました。現在のセグメント完了後に停止します。")
        else:
            self.log("再開しました。")

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
                    dropped.status = "中断"
                    dropped.message = "not started"
                except queue.Empty:
                    break

        for job in jobs:
            if job.process is not None and job.process.poll() is None:
                try:
                    job.process.kill()
                except Exception:
                    pass
        self.log("中断を送信しました。完了済みセグメントは保持します。")

    def move_finished_sources(self, profile: EncodeProfile) -> None:
        moved = 0
        kept = 0
        archive_dir = profile_archive_dir(profile)
        archive_dir.mkdir(parents=True, exist_ok=True)

        for status in scan_profile_files(profile):
            src = status.path
            missing = []
            for variant in profile.outputs:
                if not variant.enabled:
                    continue
                if not output_path_for(profile, src, variant).exists():
                    missing.append(variant.name)

            if missing:
                kept += 1
                self.log(f"Keep source: {src.name} / missing {', '.join(missing)}")
                continue

            try:
                dest = archive_dir / src.name
                if dest.exists():
                    base = dest.stem
                    suffix = dest.suffix
                    n = 1
                    while dest.exists():
                        dest = archive_dir / f"{base}_{n}{suffix}"
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
            ok = messagebox.askyesno("Exit", "ジョブが実行中です。中断して終了しますか？")
            if not ok:
                return
            app.stop_all()
            time.sleep(0.5)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
