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
        normalize_profile_gpu,
        normalize_container_extension,
        output_path_for,
        parse_ffmpeg_time,
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
        segment_file_name,
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
        normalize_profile_gpu,
        normalize_container_extension,
        output_path_for,
        parse_ffmpeg_time,
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
        segment_file_name,
        segment_ranges,
        temp_output_path_for,
        variant_by_id,
        write_concat_file,
    )
    from ffmpeg_downloader import FfmpegDownloadError, ensure_ffmpeg_available  # type: ignore


PROFILE_DIR_LABELS = {
    "input_dir": "入力先",
    "output_dir": "出力先",
    "archive_dir": "処理済みソース退避先",
}

CPU_DEVICE_LABEL = "CPU"
GPU_DEVICE_PREFIX = "GPU "
GPU_CODECS = ["hevc_nvenc", "h264_nvenc", "av1_nvenc"]
CPU_CODECS = ["libx264", "libx265"]
GPU_PRESETS = ["p1", "p2", "p3", "p4", "p5", "p6", "p7"]
CPU_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo"]
GPU_TUNES = ["none", "hq", "ll", "ull", "lossless"]
CPU_TUNES_BY_CODEC = {
    "libx264": ["none", "film", "animation", "grain", "stillimage", "fastdecode", "zerolatency", "psnr", "ssim"],
    "libx265": ["none", "psnr", "ssim", "grain", "fastdecode", "zerolatency"],
}
RATE_MODES = ["CQ", "VBR", "ABR", "CBR"]
CONTAINER_CHOICES = ["mp4", "mkv", "mov", "m4v", "webm", "ts", "m2ts"]


class SearchableCombobox(ttk.Combobox):
    def __init__(self, master: tk.Widget, values: List[str], **kwargs: object) -> None:
        self._search_values = list(values)
        super().__init__(master, values=self._search_values, **kwargs)
        self.bind("<KeyRelease>", self._filter_values, add="+")

    def _filter_values(self, event: tk.Event) -> None:
        if event.keysym in {"Up", "Down", "Left", "Right", "Return", "Escape", "Tab"}:
            return
        needle = self.get().strip().lower().lstrip(".")
        if not needle:
            self.configure(values=self._search_values)
            return
        matches = [value for value in self._search_values if needle in value.lower()]
        self.configure(values=matches or self._search_values)


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
        self.root.geometry("1180x780")
        self.root.minsize(760, 520)

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
            "bg": "#eef2f7",
            "surface": "#ffffff",
            "surface_alt": "#f8fafc",
            "text": "#111827",
            "muted": "#64748b",
            "accent": "#0f766e",
            "accent_hover": "#115e59",
            "danger": "#b91c1c",
            "log_bg": "#111827",
            "log_fg": "#d1fae5",
        }

        self.root.configure(bg=self.colors["bg"])
        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure("Surface.TFrame", background=self.colors["surface"], relief="flat")
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])
        style.configure("Surface.TLabel", background=self.colors["surface"], foreground=self.colors["text"])
        style.configure("Muted.TLabel", background=self.colors["surface"], foreground=self.colors["muted"])
        style.configure("Title.TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=self.colors["bg"], foreground=self.colors["muted"], font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=self.colors["surface"], foreground=self.colors["text"], font=("Segoe UI", 11, "bold"))
        style.configure("Accent.TButton", background=self.colors["accent"], foreground="#ffffff", font=("Segoe UI", 10, "bold"), padding=(12, 7))
        style.map("Accent.TButton", background=[("active", self.colors["accent_hover"])], foreground=[("disabled", "#e5e7eb")])
        style.configure("Danger.TButton", foreground=self.colors["danger"], font=("Segoe UI", 10, "bold"), padding=(12, 7))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 6))
        style.configure("TEntry", padding=(7, 5))
        style.configure("TCombobox", padding=(7, 5))
        style.configure("TSpinbox", padding=(7, 5))
        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background="#dde7f0", foreground=self.colors["text"], padding=(18, 9), font=("Segoe UI", 10, "bold"))
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#edf3f8")],
            foreground=[("selected", self.colors["text"])],
        )
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground=self.colors["text"], rowheight=30, font=("Segoe UI", 9), borderwidth=0)
        style.configure("Treeview.Heading", background=self.colors["surface_alt"], foreground=self.colors["text"], font=("Segoe UI", 9, "bold"), padding=(8, 6))
        style.configure("Horizontal.TProgressbar", thickness=13)

    def _surface(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=14, style="Surface.TFrame")
        return frame

    def _build_ui(self) -> None:
        self._configure_style()

        self.active_profile_var = tk.StringVar()
        self.status_var = tk.StringVar(value="待機中")
        self.progress_text_var = tk.StringVar(value="0%")
        self.input_summary_var = tk.StringVar(value="")
        self.output_summary_var = tk.StringVar(value="")

        main = ttk.Frame(self.root, padding=(18, 16), style="App.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main, style="App.TFrame")
        header.pack(fill=tk.X)
        title_block = ttk.Frame(header, style="App.TFrame")
        title_block.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title_block, text="NVEnc Archive Studio", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            title_block,
            text="プロファイル単位で入力先、出力先、処理デバイス、出力バリアントを管理します。",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        selector = ttk.Frame(header, style="App.TFrame")
        selector.pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Label(selector, text="実行プロファイル").pack(anchor=tk.W)
        self.profile_combo = ttk.Combobox(
            selector,
            textvariable=self.active_profile_var,
            width=30,
            state="readonly",
        )
        self.profile_combo.pack(anchor=tk.E, pady=(3, 0))
        self.profile_combo.bind("<<ComboboxSelected>>", self.on_profile_selected)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(14, 0))

        self.run_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.profile_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(self.run_tab, text="実行")
        self.notebook.add(self.profile_tab, text="プロファイル")
        self.run_tab_body = self._scrollable_tab(self.run_tab)
        self.profile_tab_body = self._scrollable_tab(self.profile_tab)

        self._build_run_tab()
        self._build_profile_tab()

    def _scrollable_tab(self, parent: ttk.Frame) -> ttk.Frame:
        canvas = tk.Canvas(parent, highlightthickness=0, background=self.colors["bg"])
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas, padding=14, style="App.TFrame")
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def on_content_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def event_is_inside_content(widget: tk.Misc) -> bool:
            current: Optional[tk.Misc] = widget
            while current is not None:
                if current == content:
                    return True
                current = getattr(current, "master", None)
            return False

        def on_mousewheel(event: tk.Event) -> Optional[str]:
            widget = getattr(event, "widget", None)
            if widget is None or not event_is_inside_content(widget):
                return None
            delta = int(-1 * (event.delta / 120)) if event.delta else 0
            if delta:
                canvas.yview_scroll(delta, "units")
                return "break"
            return None

        content.bind("<Configure>", on_content_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        content.bind_all("<MouseWheel>", on_mousewheel, add="+")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        return content

    def _section(self, parent: ttk.Frame, title: str) -> ttk.Frame:
        frame = self._surface(parent)
        frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(frame, text=title, style="Section.TLabel").pack(anchor=tk.W, pady=(0, 10))
        return frame

    def _row(self, parent: ttk.Frame) -> ttk.Frame:
        row = ttk.Frame(parent, style="Surface.TFrame")
        row.pack(fill=tk.X, pady=4)
        row.columnconfigure(1, weight=1)
        return row

    def _entry_row(self, parent: ttk.Frame, label: str, variable: tk.Variable, width: int = 28) -> ttk.Entry:
        row = self._row(parent)
        ttk.Label(row, text=label, width=18, style="Surface.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        entry = ttk.Entry(row, textvariable=variable, width=width)
        entry.grid(row=0, column=1, sticky="ew")
        return entry

    def _spin_row(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        from_: int,
        to: int,
    ) -> ttk.Spinbox:
        row = self._row(parent)
        ttk.Label(row, text=label, width=18, style="Surface.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        spin = ttk.Spinbox(row, from_=from_, to=to, textvariable=variable, width=8)
        spin.grid(row=0, column=1, sticky=tk.W)
        return spin

    def _combo_row(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        values: List[str],
        width: int = 28,
    ) -> ttk.Combobox:
        row = self._row(parent)
        ttk.Label(row, text=label, width=18, style="Surface.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        combo = ttk.Combobox(row, textvariable=variable, values=values, width=width, state="readonly")
        combo.grid(row=0, column=1, sticky="ew")
        return combo

    def _path_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = self._row(parent)
        ttk.Label(row, text=label, width=18, style="Surface.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        entry = ttk.Entry(row, textvariable=variable)
        entry.grid(row=0, column=1, sticky="ew")
        ttk.Button(row, text="参照", command=lambda: self.browse_dir(variable)).grid(row=0, column=2, sticky=tk.E, padx=(8, 0))

    def _build_run_tab(self) -> None:
        tab = self.run_tab_body
        summary = self._surface(tab)
        summary.pack(fill=tk.X)

        left = ttk.Frame(summary, style="Surface.TFrame")
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, textvariable=self.status_var, style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(left, textvariable=self.input_summary_var, style="Muted.TLabel").pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(left, textvariable=self.output_summary_var, style="Muted.TLabel").pack(anchor=tk.W)

        controls = ttk.Frame(summary, style="Surface.TFrame")
        controls.pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Button(controls, text="開始", style="Accent.TButton", command=self.start_current_profile).pack(side=tk.LEFT, padx=(0, 8))
        self.pause_button = ttk.Button(controls, text="一時停止", command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="停止", style="Danger.TButton", command=self.stop_all).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="保存状態から再開", command=self.resume_saved).pack(side=tk.LEFT)

        progress_box = self._surface(tab)
        progress_box.pack(fill=tk.X, pady=(12, 0))
        progress_head = ttk.Frame(progress_box, style="Surface.TFrame")
        progress_head.pack(fill=tk.X)
        ttk.Label(progress_head, text="全体進捗", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Label(progress_head, textvariable=self.progress_text_var, style="Muted.TLabel").pack(side=tk.RIGHT)
        self.overall_progress = ttk.Progressbar(progress_box, mode="determinate", maximum=100)
        self.overall_progress.pack(fill=tk.X, pady=(10, 0))

        list_box = self._surface(tab)
        list_box.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        list_head = ttk.Frame(list_box, style="Surface.TFrame")
        list_head.pack(fill=tk.X)
        ttk.Label(list_head, text="ファイル別進捗", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(list_head, text="再スキャン", command=self.scan_files).pack(side=tk.RIGHT)

        tree_frame = ttk.Frame(list_box, style="Surface.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        columns = ("file", "output", "status", "progress")
        self.progress_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10)
        for key, text, width in [
            ("file", "ファイル", 280),
            ("output", "出力", 190),
            ("status", "状態", 120),
            ("progress", "進捗", 220),
        ]:
            self.progress_tree.heading(key, text=text)
            self.progress_tree.column(key, width=width, anchor=tk.W)
        self.progress_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.progress_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.progress_tree.config(yscrollcommand=tree_scroll.set)

        log_box = self._surface(tab)
        log_box.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        log_head = ttk.Frame(log_box, style="Surface.TFrame")
        log_head.pack(fill=tk.X)
        ttk.Label(log_head, text="ログ", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(log_head, text="ログフォルダ", command=self.open_log_dir).pack(side=tk.RIGHT)
        self.log_text = ScrolledText(
            log_box,
            wrap=tk.WORD,
            height=12,
            bg=self.colors["log_bg"],
            fg=self.colors["log_fg"],
            insertbackground=self.colors["log_fg"],
            relief=tk.FLAT,
            padx=10,
            pady=10,
            font=("Consolas", 9),
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _build_profile_tab(self) -> None:
        tab = self.profile_tab_body
        form = self._surface(tab)
        form.pack(fill=tk.X)

        toolbar = ttk.Frame(form, style="Surface.TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 12))
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
        self.gpu_choice_var = tk.StringVar(value=CPU_DEVICE_LABEL)
        self.codec_var = tk.StringVar(value="hevc_nvenc")
        self.cpu_codec_var = tk.StringVar(value="libx264")
        self.preset_var = tk.StringVar(value="p7")
        self.cpu_preset_var = tk.StringVar(value="medium")
        self.tune_var = tk.StringVar(value="hq")
        self.cpu_tune_var = tk.StringVar(value="none")
        self.rate_mode_var = tk.StringVar(value="CQ")
        self.cq_var = tk.StringVar(value="18")
        self.bitrate_var = tk.StringVar(value="25000k")
        self.maxrate_var = tk.StringVar(value="40000k")
        self.bufsize_var = tk.StringVar(value="80000k")

        general = self._section(form, "基本設定")
        self._entry_row(general, "プロファイル名", self.profile_name_var)
        self._path_row(general, "入力先", self.input_dir_var)
        self._path_row(general, "出力先", self.output_dir_var)
        self._path_row(general, "処理済みソース退避先", self.archive_dir_var)

        runtime = self._section(form, "実行設定")
        self.gpu_combo = self._combo_row(runtime, "処理デバイス", self.gpu_choice_var, self._gpu_choices())
        self.gpu_combo.bind("<<ComboboxSelected>>", self.on_device_changed)
        self._spin_row(runtime, "同時実行数", self.max_jobs_var, 1, 8)
        self._spin_row(runtime, "分割間隔(分)", self.segment_minutes_var, 1, 120)

        encoder = self._section(form, "エンコード設定")
        self.gpu_settings_frame = ttk.Frame(encoder, style="Surface.TFrame")
        self.cpu_settings_frame = ttk.Frame(encoder, style="Surface.TFrame")
        self.gpu_codec_combo = self._combo_row(self.gpu_settings_frame, "GPU Codec", self.codec_var, GPU_CODECS, width=18)
        self.gpu_preset_combo = self._combo_row(self.gpu_settings_frame, "NVENC Preset", self.preset_var, GPU_PRESETS, width=12)
        self.gpu_tune_combo = self._combo_row(self.gpu_settings_frame, "NVENC Tune", self.tune_var, GPU_TUNES, width=16)
        self.cpu_codec_combo = self._combo_row(self.cpu_settings_frame, "CPU Codec", self.cpu_codec_var, CPU_CODECS, width=18)
        self.cpu_codec_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_cpu_tune_choices())
        self.cpu_preset_combo = self._combo_row(self.cpu_settings_frame, "CPU Preset", self.cpu_preset_var, CPU_PRESETS, width=18)
        self.cpu_tune_combo = self._combo_row(self.cpu_settings_frame, "CPU Tune", self.cpu_tune_var, CPU_TUNES_BY_CODEC["libx264"], width=18)

        self.rate_frame = ttk.Frame(encoder, style="Surface.TFrame")
        self.rate_frame.pack(fill=tk.X, pady=(8, 0))
        self.rate_combo = self._combo_row(self.rate_frame, "品質/レート制御", self.rate_mode_var, RATE_MODES, width=12)
        self.rate_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_rate_controls())
        self.cq_label, self.cq_entry = self._labeled_rate_entry(self.rate_frame, "CQ", self.cq_var)
        self.bitrate_label, self.bitrate_entry = self._labeled_rate_entry(self.rate_frame, "Bitrate", self.bitrate_var)
        self.maxrate_label, self.maxrate_entry = self._labeled_rate_entry(self.rate_frame, "Maxrate", self.maxrate_var)
        self.bufsize_label, self.bufsize_entry = self._labeled_rate_entry(self.rate_frame, "Bufsize", self.bufsize_var)

        outputs = self._surface(tab)
        outputs.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        output_head = ttk.Frame(outputs, style="Surface.TFrame")
        output_head.pack(fill=tk.X)
        ttk.Label(output_head, text="このプロファイルの出力バリアント", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(output_head, text="選択解除", command=self.clear_output_selection).pack(side=tk.RIGHT)

        output_body = ttk.Frame(outputs, style="Surface.TFrame")
        output_body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        tree_frame = ttk.Frame(output_body, style="Surface.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        columns = ("enabled", "name", "resolution", "folder", "container")
        self.outputs_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=7)
        for key, text, width in [
            ("enabled", "有効", 60),
            ("name", "名前", 190),
            ("resolution", "解像度", 100),
            ("folder", "フォルダ名", 190),
            ("container", "コンテナ", 90),
        ]:
            self.outputs_tree.heading(key, text=text)
            self.outputs_tree.column(key, width=width, anchor=tk.W)
        self.outputs_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.outputs_tree.yview)
        output_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.outputs_tree.config(yscrollcommand=output_scroll.set)
        self.outputs_tree.bind("<<TreeviewSelect>>", self.on_output_select)

        edit = ttk.Frame(output_body, style="Surface.TFrame")
        edit.pack(fill=tk.X, pady=(14, 0))
        edit.columnconfigure(1, weight=1)
        edit.columnconfigure(3, weight=1)
        self.output_name_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.output_resolution_var = tk.StringVar(value="1080p")
        self.output_custom_height_var = tk.StringVar(value="")
        self.output_container_var = tk.StringVar(value="mp4")
        self.output_enabled_var = tk.BooleanVar(value=True)

        ttk.Label(edit, text="名前", style="Surface.TLabel").grid(row=0, column=0, sticky=tk.W, pady=4, padx=(0, 8))
        ttk.Entry(edit, textvariable=self.output_name_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(edit, text="フォルダ名", style="Surface.TLabel").grid(row=0, column=2, sticky=tk.W, pady=4, padx=(16, 8))
        ttk.Entry(edit, textvariable=self.output_folder_var).grid(row=0, column=3, sticky="ew", pady=4)

        ttk.Label(edit, text="解像度", style="Surface.TLabel").grid(row=1, column=0, sticky=tk.W, pady=4, padx=(0, 8))
        resolution_combo = ttk.Combobox(
            edit,
            textvariable=self.output_resolution_var,
            values=list(RESOLUTION_PRESETS.keys()),
            state="readonly",
        )
        resolution_combo.grid(row=1, column=1, sticky="ew", pady=4)
        resolution_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_resolution_controls())
        self.custom_height_label = ttk.Label(edit, text="カスタム高さ", style="Surface.TLabel")
        self.custom_height_entry = ttk.Entry(edit, textvariable=self.output_custom_height_var)
        self.custom_height_label.grid(row=1, column=2, sticky=tk.W, pady=4, padx=(16, 8))
        self.custom_height_entry.grid(row=1, column=3, sticky="ew", pady=4)

        ttk.Label(edit, text="コンテナ", style="Surface.TLabel").grid(row=2, column=0, sticky=tk.W, pady=4, padx=(0, 8))
        self.output_container_combo = SearchableCombobox(
            edit,
            textvariable=self.output_container_var,
            values=CONTAINER_CHOICES,
            state="normal",
        )
        self.output_container_combo.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Checkbutton(edit, text="このバリアントを有効にする", variable=self.output_enabled_var).grid(
            row=2,
            column=2,
            columnspan=2,
            sticky=tk.W,
            pady=4,
            padx=(16, 0),
        )

        buttons = ttk.Frame(edit, style="Surface.TFrame")
        buttons.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(buttons, text="追加/更新", style="Accent.TButton", command=self.add_or_update_output).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(buttons, text="削除", command=self.remove_output).pack(side=tk.RIGHT)

        footer = ttk.Frame(tab, style="App.TFrame")
        footer.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(footer, text="FFmpegを確認/導入", command=self.download_ffmpeg_button).pack(side=tk.RIGHT)

        self.update_cpu_tune_choices()
        self.update_encoder_controls(apply_defaults=False)
        self.update_rate_controls()
        self.update_resolution_controls()

    def _labeled_rate_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar) -> tuple[ttk.Label, ttk.Entry]:
        row = self._row(parent)
        label_widget = ttk.Label(row, text=label, width=18, style="Surface.TLabel")
        label_widget.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        entry = ttk.Entry(row, textvariable=variable, width=14)
        entry.grid(row=0, column=1, sticky=tk.W)
        return label_widget, entry

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
        choices = [CPU_DEVICE_LABEL]
        choices.extend([f"{GPU_DEVICE_PREFIX}{gpu.index}: {gpu.name}" for gpu in self.gpus])
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
        self.cpu_preset_var.set(profile.cpu_preset)
        self.tune_var.set(profile.tune)
        self.cpu_tune_var.set(getattr(profile, "cpu_tune", "none"))
        self.rate_mode_var.set(profile.rate_mode)
        self.cq_var.set(str(profile.cq_value))
        self.bitrate_var.set(profile.bitrate)
        self.maxrate_var.set(profile.maxrate)
        self.bufsize_var.set(profile.bufsize)
        self.editing_outputs = [OutputVariant.from_dict(asdict(item)) for item in profile.outputs]
        self.selected_output_id = None
        self.refresh_outputs_tree()
        self.update_cpu_tune_choices()
        self.update_encoder_controls(apply_defaults=False)
        self.update_rate_controls()
        self.update_resolution_controls()

    def _choice_for_profile_gpu(self, profile: EncodeProfile) -> str:
        if not profile.use_gpu:
            return CPU_DEVICE_LABEL
        for gpu in self.gpus:
            if gpu.index == profile.gpu_index:
                return f"{GPU_DEVICE_PREFIX}{gpu.index}: {gpu.name}"
        return CPU_DEVICE_LABEL

    def _parse_gpu_choice(self) -> tuple[bool, int, str]:
        choice = self.gpu_choice_var.get()
        if not choice.startswith(GPU_DEVICE_PREFIX):
            return False, 0, ""
        prefix, _, name = choice.partition(":")
        index_text = prefix.replace(GPU_DEVICE_PREFIX.strip(), "").strip()
        if not index_text.isdigit():
            return False, 0, ""
        return True, int(index_text), name.strip()

    def on_device_changed(self, _event: object = None) -> None:
        use_gpu, _gpu_index, _gpu_name = self._parse_gpu_choice()
        self.apply_device_defaults(use_gpu)
        self.update_encoder_controls(apply_defaults=False)
        self.update_rate_controls()

    def apply_device_defaults(self, use_gpu: bool) -> None:
        self.rate_mode_var.set("CQ")
        if use_gpu:
            if self.max_jobs_var.get() < 2:
                self.max_jobs_var.set(2)
            self.cq_var.set("18")
            self.bitrate_var.set("25000k")
            self.maxrate_var.set("40000k")
            self.bufsize_var.set("80000k")
        else:
            self.max_jobs_var.set(1)
            self.cpu_codec_var.set(self.cpu_codec_var.get() or "libx264")
            self.cpu_preset_var.set(self.cpu_preset_var.get() or "medium")
            self.cpu_tune_var.set("none")
            self.cq_var.set("23")
            self.bitrate_var.set("8000k")
            self.maxrate_var.set("12000k")
            self.bufsize_var.set("24000k")
            self.update_cpu_tune_choices()

    def update_cpu_tune_choices(self) -> None:
        if not hasattr(self, "cpu_tune_combo"):
            return
        values = CPU_TUNES_BY_CODEC.get(self.cpu_codec_var.get(), CPU_TUNES_BY_CODEC["libx264"])
        self.cpu_tune_combo.configure(values=values)
        if self.cpu_tune_var.get() not in values:
            self.cpu_tune_var.set("none")

    def update_encoder_controls(self, apply_defaults: bool = False) -> None:
        if not hasattr(self, "gpu_settings_frame"):
            return
        use_gpu, _gpu_index, _gpu_name = self._parse_gpu_choice()
        if apply_defaults:
            self.apply_device_defaults(use_gpu)
        if use_gpu:
            self.cpu_settings_frame.pack_forget()
            self.gpu_settings_frame.pack(fill=tk.X, before=self.rate_frame)
        else:
            self.gpu_settings_frame.pack_forget()
            self.cpu_settings_frame.pack(fill=tk.X, before=self.rate_frame)

        if hasattr(self, "cq_label"):
            self.cq_label.configure(text="CQ" if use_gpu else "CRF")

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
        if not hasattr(self, "custom_height_entry"):
            return
        if self.output_resolution_var.get() == "Custom":
            self.custom_height_label.grid()
            self.custom_height_entry.grid()
            self.custom_height_entry.configure(state=tk.NORMAL)
        else:
            self.output_custom_height_var.set("")
            self.custom_height_entry.configure(state=tk.DISABLED)
            self.custom_height_label.grid_remove()
            self.custom_height_entry.grid_remove()

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
        if not any(item.enabled for item in self.editing_outputs):
            messagebox.showerror("入力エラー", "出力バリアントを1つ以上有効にしてください。")
            return None

        input_dir = self.input_dir_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        archive_dir = self.archive_dir_var.get().strip()
        if not input_dir or not output_dir or not archive_dir:
            messagebox.showerror("入力エラー", "入力先、出力先、処理済みソース退避先を入力してください。")
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
            cpu_preset=self.cpu_preset_var.get(),
            tune=self.tune_var.get(),
            cpu_tune=self.cpu_tune_var.get(),
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
                values=("有効" if variant.enabled else "無効", variant.name, resolution, variant.folder_name, variant.container),
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
                if hasattr(self, "output_enabled_var"):
                    self.output_enabled_var.set(variant.enabled)
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
        if hasattr(self, "output_enabled_var"):
            self.output_enabled_var.set(True)
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
            enabled=self.output_enabled_var.get() if hasattr(self, "output_enabled_var") else True,
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
        files = self.scan_profile_files_or_show_error(profile)
        if files is None:
            self.files = []
            self.render_scan_rows(profile)
            return
        self.files = files
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
        if not any(variant.enabled for variant in profile.outputs):
            messagebox.showerror("入力エラー", "出力バリアントを1つ以上有効にしてください。")
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

    def ensure_profile_dirs_or_show_error(self, profile: EncodeProfile) -> bool:
        try:
            ensure_profile_dirs(profile)
        except OSError as exc:
            messagebox.showerror("フォルダ作成エラー", f"入出力フォルダを作成できませんでした。\n{exc}")
            self.log(f"Profile folder setup failed: {exc}")
            return False
        return True

    def scan_profile_files_or_show_error(self, profile: EncodeProfile) -> Optional[List[FileStatus]]:
        try:
            return scan_profile_files(profile)
        except OSError as exc:
            messagebox.showerror("フォルダ読み込みエラー", f"入力フォルダを読み込めませんでした。\n{exc}")
            self.log(f"Profile scan failed: {exc}")
            return None

    def start_current_profile(self) -> None:
        profile = normalize_profile_gpu(self.current_profile(), self.gpus)
        if not self.validate_profile_before_run(profile):
            return

        if not self.ensure_profile_dirs_or_show_error(profile):
            return
        files = self.scan_profile_files_or_show_error(profile)
        if files is None:
            return
        self.files = files
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

        profile = normalize_profile_gpu(profile_from_state(data, self.paths, self.gpus), self.gpus)
        if not self.validate_profile_before_run(profile):
            return
        if not self.ensure_profile_dirs_or_show_error(profile):
            return
        specs = resumable_specs(profile, data)
        if not specs:
            clear_state(self.paths)
            self.log("保存状態はすでに完了済みでした。状態ファイルを削除しました。")
            self.scan_files()
            return

        if not self.ensure_ffmpeg_before_run():
            return

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
        # job_key includes source size/mtime, so keep one segment root for this run.
        segment_root = segment_dir_for(self.paths, src, job.profile, job.variant)
        self.log(f"Start {job.variant.name}: {src.name}")
        self.log(f"Log file: {job.log_file}")

        segment_root.mkdir(parents=True, exist_ok=True)

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
                    final_segment = segment_root / segment_file_name(job.variant, index)
                    partial_segment = segment_root / segment_file_name(job.variant, index, partial=True)
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

                concat_file = segment_root / "concat.txt"
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
                shutil.rmtree(segment_root, ignore_errors=True)
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
        with self.lock:
            job.process = process
            stop_requested = self.stop_requested
        if stop_requested and process.poll() is None:
            process.kill()

        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    if line:
                        log_fp.write(line + "\n")
                        log_fp.flush()
                        self.log(f"job {job.job_id}: {line}")
                        self._update_job_progress_from_line(job, line, segment_duration)

            return process.wait()
        finally:
            with self.lock:
                if job.process is process:
                    job.process = None

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
            with self.lock:
                process = job.process
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass
        self.log("中断を送信しました。完了済みセグメントは保持します。")

    def move_finished_sources(self, profile: EncodeProfile) -> None:
        moved = 0
        kept = 0
        archive_dir = profile_archive_dir(profile)
        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
            statuses = scan_profile_files(profile)
        except OSError as exc:
            self.log(f"Move skipped: profile folder access failed / {exc}")
            return

        for status in statuses:
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
