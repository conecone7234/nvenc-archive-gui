from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    from ffmpeg_nvenc_gui.core import (
        AppPaths,
        BACKEND_AMF,
        BACKEND_CPU,
        BACKEND_NVENC,
        BACKEND_QSV,
        CPU_RESOURCE_ID,
        EncodeProfile,
        FileStatus,
        GpuInfo,
        HardwareResource,
        JobSpec,
        OutputVariant,
        RESOLUTION_PRESETS,
        build_audio_command,
        build_concat_command,
        build_ffmpeg_command,
        build_job_specs,
        build_mux_command,
        build_paths,
        clear_state,
        command_to_text,
        default_profile,
        detect_nvidia_gpus,
        duplicate_output_targets,
        duplicate_output_targets_for_outputs,
        encoder_codec,
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
        probe_has_audio,
        probe_duration,
        profile_archive_dir,
        profile_from_state,
        profile_input_dir,
        resumable_specs,
        safe_folder_name,
        save_profiles,
        save_state,
        scan_profile_files,
        joined_video_path_for,
        segment_dir_for,
        segment_file_name,
        segment_ranges,
        temp_audio_path_for,
        temp_output_path_for,
        resource_backend,
        resource_index,
        variant_by_id,
        variant_resource_ids,
        write_concat_file,
    )
    from ffmpeg_nvenc_gui.ffmpeg_downloader import (
        FfmpegDownloadError,
        encoder_capabilities,
        ensure_ffmpeg_available,
        smoke_test_encoder,
    )
except ModuleNotFoundError:
    from core import (  # type: ignore
        AppPaths,
        BACKEND_AMF,
        BACKEND_CPU,
        BACKEND_NVENC,
        BACKEND_QSV,
        CPU_RESOURCE_ID,
        EncodeProfile,
        FileStatus,
        GpuInfo,
        HardwareResource,
        JobSpec,
        OutputVariant,
        RESOLUTION_PRESETS,
        build_audio_command,
        build_concat_command,
        build_ffmpeg_command,
        build_job_specs,
        build_mux_command,
        build_paths,
        clear_state,
        command_to_text,
        default_profile,
        detect_nvidia_gpus,
        duplicate_output_targets,
        duplicate_output_targets_for_outputs,
        encoder_codec,
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
        probe_has_audio,
        probe_duration,
        profile_archive_dir,
        profile_from_state,
        profile_input_dir,
        resumable_specs,
        safe_folder_name,
        save_profiles,
        save_state,
        scan_profile_files,
        joined_video_path_for,
        segment_dir_for,
        segment_file_name,
        segment_ranges,
        temp_audio_path_for,
        temp_output_path_for,
        resource_backend,
        resource_index,
        variant_by_id,
        variant_resource_ids,
        write_concat_file,
    )
    from ffmpeg_downloader import (  # type: ignore
        FfmpegDownloadError,
        encoder_capabilities,
        ensure_ffmpeg_available,
        smoke_test_encoder,
    )


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


def select_compatible_resource_ids(allowed_ids: List[str], selected_ids: List[str]) -> List[str]:
    selected = [resource_id for resource_id in selected_ids if resource_id in allowed_ids]
    return selected or allowed_ids[:1]


def backend_allows_cq(backend: str) -> bool:
    return backend in {BACKEND_CPU, BACKEND_NVENC}


def backend_accepts_rate_mode(backend: str, rate_mode: str) -> bool:
    return rate_mode.upper() != "CQ" or backend_allows_cq(backend)


def rate_modes_for_backend(backend: str) -> List[str]:
    if backend_allows_cq(backend):
        return RATE_MODES
    return [mode for mode in RATE_MODES if mode != "CQ"]


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
    resource_id: str = ""
    resource_slot: int = 0
    resource_slots_reserved: int = 1
    resource_slot_indexes: List[int] = field(default_factory=list)
    process: Optional[subprocess.Popen] = None
    processes: Dict[int, subprocess.Popen] = field(default_factory=dict)
    status: str = "waiting"
    message: str = ""
    total_segments: int = 1
    completed_segments: int = 0
    current_segment: int = 0
    progress: float = 0.0
    segment_progress: Dict[int, float] = field(default_factory=dict)
    completed_segment_indexes: set[int] = field(default_factory=set)


class EncoderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NVEnc Archive Studio")
        self.root.geometry("1180x720")
        self.root.minsize(980, 620)

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
        self.active_resource_slots: Dict[str, int] = {}
        self.active_resource_slot_indexes: Dict[str, set[int]] = {}
        self.encoder_capabilities: Dict[str, Dict[str, object]] = {}

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
            text="設定で入力先と出力先を定義し、出力プロファイルごとにエンコード内容を切り替えます。",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        selector = ttk.Frame(header, style="App.TFrame")
        selector.pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Label(selector, text="実行設定").pack(anchor=tk.W)
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
        self.notebook.add(self.profile_tab, text="設定")
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
        ttk.Label(toolbar, text="設定", style="Section.TLabel").pack(side=tk.LEFT)
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
        self.resource_enabled_vars: Dict[str, tk.BooleanVar] = {}
        self.resource_slot_vars: Dict[str, tk.IntVar] = {}
        self.output_resource_vars: Dict[str, tk.BooleanVar] = {}

        general = self._section(form, "基本設定")
        self._entry_row(general, "設定名", self.profile_name_var)
        self._path_row(general, "入力先", self.input_dir_var)
        self._path_row(general, "出力先", self.output_dir_var)
        self._path_row(general, "処理済みソース退避先", self.archive_dir_var)

        runtime = self._section(form, "実行設定")
        ttk.Label(runtime, text="Processing resources", style="Surface.TLabel").pack(anchor=tk.W)
        self.resource_frame = ttk.Frame(runtime, style="Surface.TFrame")
        self.resource_frame.pack(fill=tk.X, pady=(4, 8))
        self._spin_row(runtime, "並列セグメント数", self.max_jobs_var, 1, 8)
        self._spin_row(runtime, "分割間隔(分)", self.segment_minutes_var, 1, 120)

        outputs = self._surface(tab)
        outputs.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        output_head = ttk.Frame(outputs, style="Surface.TFrame")
        output_head.pack(fill=tk.X)
        ttk.Label(output_head, text="出力プロファイル", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(output_head, text="選択を解除", command=self.clear_output_selection).pack(side=tk.RIGHT)

        output_body = ttk.Frame(outputs, style="Surface.TFrame")
        output_body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        tree_frame = ttk.Frame(output_body, style="Surface.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        columns = ("enabled", "name", "backend", "encoder", "resolution", "folder", "container")
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
        for key, text, width in [("backend", "Backend", 90), ("encoder", "Encoder", 120)]:
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
        self.output_filename_template_var = tk.StringVar(value="{source}")
        self.output_backend_var = tk.StringVar(value=BACKEND_NVENC if self.gpus else BACKEND_CPU)
        self.output_encoder_var = tk.StringVar(value="hevc_nvenc" if self.gpus else "libx264")
        self.output_split_encode_mode_var = tk.StringVar(value="auto")
        self.output_gpu_choice_var = tk.StringVar(value=CPU_DEVICE_LABEL)
        self.output_codec_var = tk.StringVar(value="hevc_nvenc")
        self.output_cpu_codec_var = tk.StringVar(value="libx264")
        self.output_preset_var = tk.StringVar(value="p7")
        self.output_cpu_preset_var = tk.StringVar(value="medium")
        self.output_cpu_tune_var = tk.StringVar(value="none")
        self.output_tune_var = tk.StringVar(value="hq")
        self.output_rate_mode_var = tk.StringVar(value="CQ")
        self.output_cq_var = tk.StringVar(value="18")
        self.output_bitrate_var = tk.StringVar(value="25000k")
        self.output_maxrate_var = tk.StringVar(value="40000k")
        self.output_bufsize_var = tk.StringVar(value="80000k")
        self.output_pix_fmt_var = tk.StringVar(value="nv12")
        self.output_scale_flags_var = tk.StringVar(value="lanczos+accurate_rnd")
        self.output_audio_codec_var = tk.StringVar(value="copy")
        self.output_audio_bitrate_var = tk.StringVar(value="")
        self.output_audio_container_var = tk.StringVar(value="")
        self.output_extra_input_args_var = tk.StringVar(value="")
        self.output_extra_video_args_var = tk.StringVar(value="")
        self.output_extra_audio_args_var = tk.StringVar(value="")
        self.output_extra_output_args_var = tk.StringVar(value="")
        self.output_extra_concat_args_var = tk.StringVar(value="")
        self.output_extra_mux_args_var = tk.StringVar(value="")

        def grid_entry(label: str, variable: tk.Variable, row: int, pair: int = 0) -> ttk.Entry:
            label_column = pair * 2
            ttk.Label(edit, text=label, style="Surface.TLabel").grid(
                row=row,
                column=label_column,
                sticky=tk.W,
                pady=4,
                padx=(0 if pair == 0 else 16, 8),
            )
            entry = ttk.Entry(edit, textvariable=variable)
            entry.grid(row=row, column=label_column + 1, sticky="ew", pady=4)
            return entry

        def grid_combo(
            label: str,
            variable: tk.Variable,
            values: List[str],
            row: int,
            pair: int = 0,
            width: int = 18,
        ) -> ttk.Combobox:
            label_column = pair * 2
            ttk.Label(edit, text=label, style="Surface.TLabel").grid(
                row=row,
                column=label_column,
                sticky=tk.W,
                pady=4,
                padx=(0 if pair == 0 else 16, 8),
            )
            combo = ttk.Combobox(edit, textvariable=variable, values=values, width=width, state="readonly")
            combo.grid(row=row, column=label_column + 1, sticky="ew", pady=4)
            return combo

        grid_entry("名前", self.output_name_var, 0, 0)
        grid_entry("フォルダ名", self.output_folder_var, 0, 1)
        grid_entry("ファイル名テンプレート", self.output_filename_template_var, 1, 0)
        ttk.Checkbutton(edit, text="このプロファイルを有効にする", variable=self.output_enabled_var).grid(
            row=1,
            column=2,
            columnspan=2,
            sticky=tk.W,
            pady=4,
            padx=(16, 0),
        )

        ttk.Label(edit, text="解像度", style="Surface.TLabel").grid(row=2, column=0, sticky=tk.W, pady=4, padx=(0, 8))
        resolution_combo = ttk.Combobox(
            edit,
            textvariable=self.output_resolution_var,
            values=list(RESOLUTION_PRESETS.keys()),
            state="readonly",
        )
        resolution_combo.grid(row=2, column=1, sticky="ew", pady=4)
        resolution_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_resolution_controls())
        self.custom_height_label = ttk.Label(edit, text="カスタム高さ", style="Surface.TLabel")
        self.custom_height_entry = ttk.Entry(edit, textvariable=self.output_custom_height_var)
        self.custom_height_label.grid(row=2, column=2, sticky=tk.W, pady=4, padx=(16, 8))
        self.custom_height_entry.grid(row=2, column=3, sticky="ew", pady=4)

        ttk.Label(edit, text="コンテナ", style="Surface.TLabel").grid(row=3, column=0, sticky=tk.W, pady=4, padx=(0, 8))
        self.output_container_combo = SearchableCombobox(
            edit,
            textvariable=self.output_container_var,
            values=CONTAINER_CHOICES,
            state="normal",
        )
        self.output_container_combo.grid(row=3, column=1, sticky="ew", pady=4)
        self.output_backend_combo = grid_combo("Backend", self.output_backend_var, [BACKEND_CPU, BACKEND_NVENC, BACKEND_QSV, BACKEND_AMF], 3, 1, width=14)
        self.output_backend_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_output_encoder_controls())
        self.output_encoder_combo = grid_combo("Encoder", self.output_encoder_var, self._encoders_for_backend(self.output_backend_var.get()), 4, 0, width=18)
        self.output_encoder_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_output_encoder_controls())
        self.output_preset_combo = grid_combo("NVENC Preset", self.output_preset_var, GPU_PRESETS, 4, 1)
        self.output_tune_combo = grid_combo("NVENC Tune", self.output_tune_var, GPU_TUNES, 5, 0)
        self.output_split_combo = grid_combo("SFE", self.output_split_encode_mode_var, ["auto"], 5, 1, width=18)
        self.output_rate_combo = grid_combo("Rate", self.output_rate_mode_var, RATE_MODES, 6, 0)
        self.output_rate_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_output_rate_controls())
        self.output_cpu_codec_combo = grid_combo("CPU Codec", self.output_cpu_codec_var, CPU_CODECS, 6, 1)
        self.output_cpu_codec_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_output_cpu_tune_choices())
        self.output_cpu_preset_combo = grid_combo("CPU Preset", self.output_cpu_preset_var, CPU_PRESETS, 7, 0)
        self.output_cpu_tune_combo = grid_combo("CPU Tune", self.output_cpu_tune_var, CPU_TUNES_BY_CODEC["libx264"], 7, 1)
        ttk.Label(edit, text="Resources", style="Surface.TLabel").grid(row=8, column=0, sticky=tk.W, pady=4, padx=(0, 8))
        self.output_resource_frame = ttk.Frame(edit, style="Surface.TFrame")
        self.output_resource_frame.grid(row=8, column=1, columnspan=3, sticky="ew", pady=4)
        self.output_cq_entry = grid_entry("CQ/CRF", self.output_cq_var, 9, 0)
        self.output_bitrate_entry = grid_entry("Bitrate", self.output_bitrate_var, 9, 1)
        self.output_maxrate_entry = grid_entry("Maxrate", self.output_maxrate_var, 10, 0)
        self.output_bufsize_entry = grid_entry("Bufsize", self.output_bufsize_var, 10, 1)
        grid_entry("Pix fmt", self.output_pix_fmt_var, 11, 0)
        grid_entry("Scale flags", self.output_scale_flags_var, 11, 1)
        grid_entry("Audio codec", self.output_audio_codec_var, 12, 0)
        grid_entry("Audio bitrate", self.output_audio_bitrate_var, 12, 1)
        grid_entry("Audio container", self.output_audio_container_var, 13, 0)
        grid_entry("FFmpeg input args", self.output_extra_input_args_var, 13, 1)
        grid_entry("FFmpeg video args", self.output_extra_video_args_var, 14, 0)
        grid_entry("FFmpeg audio args", self.output_extra_audio_args_var, 14, 1)
        grid_entry("FFmpeg output args", self.output_extra_output_args_var, 15, 0)
        grid_entry("FFmpeg concat args", self.output_extra_concat_args_var, 15, 1)
        grid_entry("FFmpeg mux args", self.output_extra_mux_args_var, 16, 0)

        buttons = ttk.Frame(edit, style="Surface.TFrame")
        buttons.grid(row=17, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(buttons, text="追加/更新", style="Accent.TButton", command=self.add_or_update_output).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(buttons, text="削除", command=self.remove_output).pack(side=tk.RIGHT)

        self.update_cpu_tune_choices()
        self.update_output_cpu_tune_choices()
        self.update_encoder_controls(apply_defaults=False)
        self.update_rate_controls()
        self.update_output_rate_controls()
        self.update_resolution_controls()
        self.update_output_encoder_controls()

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

    def render_resource_controls(self, profile: EncodeProfile) -> None:
        if not hasattr(self, "resource_frame"):
            return
        for child in self.resource_frame.winfo_children():
            child.destroy()
        self.resource_enabled_vars = {}
        self.resource_slot_vars = {}
        selected = set(profile.resource_ids or [CPU_RESOURCE_ID])
        for row, resource in enumerate(profile.hardware_resources):
            enabled_var = tk.BooleanVar(value=resource.id in selected)
            slot_var = tk.IntVar(value=max(1, resource.concurrency_slots))
            self.resource_enabled_vars[resource.id] = enabled_var
            self.resource_slot_vars[resource.id] = slot_var
            ttk.Checkbutton(
                self.resource_frame,
                text=f"{resource.label} ({resource.backend})",
                variable=enabled_var,
                command=self.update_output_encoder_controls,
            ).grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Label(self.resource_frame, text="slots", style="Surface.TLabel").grid(row=row, column=1, sticky=tk.W, padx=(16, 4))
            ttk.Spinbox(self.resource_frame, from_=1, to=16, textvariable=slot_var, width=5).grid(row=row, column=2, sticky=tk.W, pady=2)
            if resource.detected_encoder_engines:
                detail = f"detected NVENC: {resource.detected_encoder_engines}"
            elif resource.detection_error:
                detail = "manual fallback"
            else:
                detail = ""
            if detail:
                ttk.Label(self.resource_frame, text=detail, style="Muted.TLabel").grid(row=row, column=3, sticky=tk.W, padx=(10, 0))
        self.update_output_encoder_controls()

    def collect_hardware_resources_from_form(self, current: EncodeProfile) -> List[HardwareResource]:
        resources: List[HardwareResource] = []
        for resource in current.hardware_resources:
            slots_var = self.resource_slot_vars.get(resource.id)
            slots = slots_var.get() if slots_var is not None else resource.concurrency_slots
            resources.append(
                HardwareResource(
                    id=resource.id,
                    label=resource.label,
                    kind=resource.kind,
                    backend=resource.backend,
                    vendor=resource.vendor,
                    index=resource.index,
                    concurrency_slots=slots,
                    detected_encoder_engines=resource.detected_encoder_engines,
                    detection_error=resource.detection_error,
                )
            )
        return resources

    def selected_profile_resource_ids(self) -> List[str]:
        selected = [
            resource_id
            for resource_id, var in self.resource_enabled_vars.items()
            if var.get()
        ]
        return selected or [CPU_RESOURCE_ID]

    def _encoders_for_backend(self, backend: str) -> List[str]:
        backend = backend or BACKEND_CPU
        if self.encoder_capabilities:
            encoders = [
                name
                for name in self.encoder_capabilities
                if (
                    (backend == BACKEND_CPU and name in CPU_CODECS)
                    or (backend == BACKEND_NVENC and name.endswith("_nvenc"))
                    or (backend == BACKEND_QSV and name.endswith("_qsv"))
                    or (backend == BACKEND_AMF and name.endswith("_amf"))
                )
            ]
            if encoders:
                return sorted(encoders)
        if backend == BACKEND_NVENC:
            return GPU_CODECS
        if backend == BACKEND_QSV:
            return ["h264_qsv", "hevc_qsv", "av1_qsv"]
        if backend == BACKEND_AMF:
            return ["h264_amf", "hevc_amf", "av1_amf"]
        return CPU_CODECS

    def selected_output_resource_ids(self) -> List[str]:
        selected = [
            resource_id
            for resource_id, var in self.output_resource_vars.items()
            if var.get()
        ]
        return selected

    def split_encode_modes_for_encoder(self, encoder: str) -> List[str]:
        caps = self.encoder_capabilities.get(encoder, {})
        if not bool(caps.get("supports_split_encode_mode")):
            return []
        detected_modes = caps.get("split_encode_modes", [])
        modes: List[str] = []
        if isinstance(detected_modes, list):
            for item in detected_modes:
                value = str(item or "").strip().lower()
                if value and value not in modes:
                    modes.append(value)
        if "auto" not in modes:
            modes.insert(0, "auto")
        if "disabled" not in modes:
            modes.append("disabled")
        return modes

    def render_output_resource_controls(self, profile: EncodeProfile, backend: str, selected: List[str]) -> None:
        if not hasattr(self, "output_resource_frame"):
            return
        for child in self.output_resource_frame.winfo_children():
            child.destroy()
        self.output_resource_vars = {}
        active_profile_resources = set(self.selected_profile_resource_ids())
        allowed = [
            resource
            for resource in profile.hardware_resources
            if resource.id in active_profile_resources and resource_backend(resource.id) == backend
        ]
        selected = select_compatible_resource_ids([resource.id for resource in allowed], selected)
        for index, resource in enumerate(allowed):
            var = tk.BooleanVar(value=resource.id in selected)
            self.output_resource_vars[resource.id] = var
            ttk.Checkbutton(
                self.output_resource_frame,
                text=resource.label,
                variable=var,
            ).grid(row=0, column=index, sticky=tk.W, padx=(0 if index == 0 else 12, 0))
        if not allowed:
            ttk.Label(self.output_resource_frame, text="No compatible resource enabled", style="Muted.TLabel").grid(row=0, column=0, sticky=tk.W)

    def update_output_encoder_controls(self) -> None:
        if not hasattr(self, "output_encoder_combo"):
            return
        backend = self.output_backend_var.get() or BACKEND_CPU
        encoders = self._encoders_for_backend(backend)
        self.output_encoder_combo.configure(values=encoders)
        if self.output_encoder_var.get() not in encoders:
            self.output_encoder_var.set(encoders[0] if encoders else "")

        encoder = self.output_encoder_var.get()
        self.output_codec_var.set(encoder if backend == BACKEND_NVENC else self.output_codec_var.get())
        self.output_cpu_codec_var.set(encoder if backend == BACKEND_CPU else self.output_cpu_codec_var.get())
        rate_modes = rate_modes_for_backend(backend)
        if hasattr(self, "output_rate_combo"):
            self.output_rate_combo.configure(values=rate_modes)
        if self.output_rate_mode_var.get().upper() not in rate_modes:
            self.output_rate_mode_var.set(rate_modes[0])
            self.update_output_rate_controls()

        nvenc_state = tk.NORMAL if backend == BACKEND_NVENC else tk.DISABLED
        cpu_state = tk.NORMAL if backend == BACKEND_CPU else tk.DISABLED
        for widget in (getattr(self, "output_preset_combo", None), getattr(self, "output_tune_combo", None)):
            if widget is not None:
                widget.configure(state=nvenc_state)
        for widget in (
            getattr(self, "output_cpu_codec_combo", None),
            getattr(self, "output_cpu_preset_combo", None),
            getattr(self, "output_cpu_tune_combo", None),
        ):
            if widget is not None:
                widget.configure(state=cpu_state)

        split_modes = self.split_encode_modes_for_encoder(encoder)
        if backend == BACKEND_NVENC and encoder in {"hevc_nvenc", "av1_nvenc"} and split_modes:
            self.output_split_combo.configure(values=split_modes, state=tk.NORMAL)
        else:
            split_modes = ["auto"]
            self.output_split_encode_mode_var.set("auto")
            self.output_split_combo.configure(values=["auto"], state=tk.DISABLED)
        if self.output_split_encode_mode_var.get() not in split_modes:
            self.output_split_encode_mode_var.set(split_modes[0])

        selected = self.selected_output_resource_ids()
        self.render_output_resource_controls(self.current_profile(), backend, selected)

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
        self.render_resource_controls(profile)
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
        self.set_output_edit_defaults(profile)
        self.refresh_outputs_tree()
        self.update_cpu_tune_choices()
        self.update_output_cpu_tune_choices()
        self.update_encoder_controls(apply_defaults=False)
        self.update_rate_controls()
        self.update_resolution_controls()
        self.update_output_encoder_controls()

    def _choice_for_profile_gpu(self, profile: EncodeProfile) -> str:
        if not profile.use_gpu:
            return CPU_DEVICE_LABEL
        for gpu in self.gpus:
            if gpu.index == profile.gpu_index:
                return f"{GPU_DEVICE_PREFIX}{gpu.index}: {gpu.name}"
        return CPU_DEVICE_LABEL

    def _parse_gpu_choice(self) -> tuple[bool, int, str]:
        return self._parse_gpu_choice_value(self.gpu_choice_var.get())

    def _parse_output_gpu_choice(self) -> tuple[bool, int, str]:
        return self._parse_gpu_choice_value(self.output_gpu_choice_var.get())

    def _parse_gpu_choice_value(self, choice: str) -> tuple[bool, int, str]:
        if not choice.startswith(GPU_DEVICE_PREFIX):
            return False, 0, ""
        prefix, _, name = choice.partition(":")
        index_text = prefix.replace(GPU_DEVICE_PREFIX.strip(), "").strip()
        if not index_text.isdigit():
            return False, 0, ""
        return True, int(index_text), name.strip()

    def _choice_for_variant_gpu(self, profile: EncodeProfile, variant: OutputVariant) -> str:
        use_gpu = profile.use_gpu if variant.use_gpu is None else variant.use_gpu
        if not use_gpu:
            return CPU_DEVICE_LABEL
        gpu_index = profile.gpu_index if variant.gpu_index is None else variant.gpu_index
        for gpu in self.gpus:
            if gpu.index == gpu_index:
                return f"{GPU_DEVICE_PREFIX}{gpu.index}: {gpu.name}"
        return CPU_DEVICE_LABEL

    def set_output_edit_defaults(self, profile: EncodeProfile) -> None:
        self.output_name_var.set("")
        self.output_folder_var.set("")
        self.output_resolution_var.set("1080p")
        self.output_custom_height_var.set("")
        self.output_container_var.set("mp4")
        self.output_enabled_var.set(True)
        self.output_filename_template_var.set("{source}")
        default_backend = BACKEND_NVENC if any(resource_backend(item) == BACKEND_NVENC for item in profile.resource_ids) else BACKEND_CPU
        self.output_backend_var.set(default_backend)
        self.output_encoder_var.set("hevc_nvenc" if default_backend == BACKEND_NVENC else "libx264")
        self.output_split_encode_mode_var.set("auto")
        self.output_gpu_choice_var.set(self._choice_for_profile_gpu(profile))
        self.output_codec_var.set(profile.codec)
        self.output_cpu_codec_var.set(profile.cpu_codec)
        self.output_preset_var.set(profile.preset)
        self.output_cpu_preset_var.set(profile.cpu_preset)
        self.output_cpu_tune_var.set(getattr(profile, "cpu_tune", "none"))
        self.output_tune_var.set(profile.tune)
        self.output_rate_mode_var.set(profile.rate_mode)
        self.output_cq_var.set(str(profile.cq_value))
        self.output_bitrate_var.set(profile.bitrate)
        self.output_maxrate_var.set(profile.maxrate)
        self.output_bufsize_var.set(profile.bufsize)
        self.output_pix_fmt_var.set(profile.pix_fmt)
        self.output_scale_flags_var.set(profile.scale_flags)
        self.output_audio_codec_var.set("copy")
        self.output_audio_bitrate_var.set("")
        self.output_audio_container_var.set("")
        self.output_extra_input_args_var.set("")
        self.output_extra_video_args_var.set("")
        self.output_extra_audio_args_var.set("")
        self.output_extra_output_args_var.set("")
        self.output_extra_concat_args_var.set("")
        self.output_extra_mux_args_var.set("")
        self.update_output_cpu_tune_choices()
        self.update_output_rate_controls()
        self.update_resolution_controls()
        self.render_output_resource_controls(profile, default_backend, [])
        self.update_output_encoder_controls()

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

    def update_output_cpu_tune_choices(self) -> None:
        if not hasattr(self, "output_cpu_tune_combo"):
            return
        values = CPU_TUNES_BY_CODEC.get(self.output_cpu_codec_var.get(), CPU_TUNES_BY_CODEC["libx264"])
        self.output_cpu_tune_combo.configure(values=values)
        if self.output_cpu_tune_var.get() not in values:
            self.output_cpu_tune_var.set("none")

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

    @staticmethod
    def _cq_value_for_rate_mode(rate_mode: str, raw_value: str, fallback: int) -> Optional[int]:
        if rate_mode.upper() not in {"CQ", "VBR"}:
            return fallback
        try:
            return int(raw_value)
        except ValueError:
            return None

    def _output_cq_fallback(self) -> int:
        for variant in self.editing_outputs:
            if variant.id == self.selected_output_id and variant.cq_value is not None:
                return variant.cq_value
        try:
            return int(self.cq_var.get())
        except ValueError:
            pass
        return self.current_profile().cq_value

    def update_rate_controls(self) -> None:
        if not hasattr(self, "cq_entry"):
            return
        mode = self.rate_mode_var.get().upper()
        controls = {
            self.cq_entry: mode in {"CQ", "VBR"},
            self.bitrate_entry: mode in {"VBR", "ABR", "CBR"},
            self.maxrate_entry: mode == "VBR",
            self.bufsize_entry: mode in {"VBR", "CBR"},
        }
        for widget, enabled in controls.items():
            widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def update_output_rate_controls(self) -> None:
        if not hasattr(self, "output_cq_entry"):
            return
        mode = self.output_rate_mode_var.get().upper()
        controls = {
            self.output_cq_entry: mode in {"CQ", "VBR"},
            self.output_bitrate_entry: mode in {"VBR", "ABR", "CBR"},
            self.output_maxrate_entry: mode == "VBR",
            self.output_bufsize_entry: mode in {"VBR", "CBR"},
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
            messagebox.showerror("入力エラー", "並列セグメント数と分割間隔は数値で入力してください。")
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
            messagebox.showerror("入力エラー", "並列セグメント数は1以上にしてください。")
            return None
        if segment_minutes < 1:
            messagebox.showerror("入力エラー", "分割間隔は1分以上にしてください。")
            return None
        if not any(item.enabled for item in self.editing_outputs):
            messagebox.showerror("入力エラー", "出力プロファイルを1つ以上有効にしてください。")
            return None

        input_dir = self.input_dir_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        archive_dir = self.archive_dir_var.get().strip()
        if not input_dir or not output_dir or not archive_dir:
            messagebox.showerror("入力エラー", "入力先、出力先、処理済みソース退避先を入力してください。")
            return None

        hardware_resources = self.collect_hardware_resources_from_form(current)
        profile_resource_ids = self.selected_profile_resource_ids()
        if not profile_resource_ids:
            messagebox.showerror("入力エラー", "処理リソースを1つ以上有効にしてください。")
            return None
        use_gpu = any(resource_backend(resource_id) != BACKEND_CPU for resource_id in profile_resource_ids)
        first_gpu = next((resource_id for resource_id in profile_resource_ids if resource_backend(resource_id) == BACKEND_NVENC), "")
        gpu_index = resource_index(first_gpu) if first_gpu else 0
        gpu_name = ""
        for resource in hardware_resources:
            if resource.id == first_gpu:
                gpu_name = resource.label
                break
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
            resource_ids=profile_resource_ids,
            hardware_resources=hardware_resources,
            outputs=[OutputVariant.from_dict(asdict(item)) for item in self.editing_outputs],
        )
        for variant in profile.outputs:
            if not variant.enabled:
                continue
            resources = variant_resource_ids(profile, variant)
            if not resources:
                messagebox.showerror("入力エラー", f"{variant.name}: 使用可能なリソースを選択してください。")
                return None
            expected_backend = variant.backend or resource_backend(resources[0])
            if any(resource_backend(resource_id) != expected_backend for resource_id in resources):
                messagebox.showerror("入力エラー", f"{variant.name}: backendの異なるリソースは混在できません。")
                return None
            rate_mode = (variant.rate_mode or profile.rate_mode or "CQ").upper()
            if not backend_accepts_rate_mode(expected_backend, rate_mode):
                messagebox.showerror("入力エラー", f"{variant.name}: CQ は CPU/NVENC のみで使用できます。QSV/AMF では VBR/ABR/CBR を選択してください。")
                return None
            missing = missing_rate_fields(profile, variant)
            if missing:
                labels = ", ".join(missing)
                messagebox.showerror("入力エラー", f"{variant.name}: {labels} を入力してください。")
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
            backend = variant.backend or (BACKEND_NVENC if variant.use_gpu else BACKEND_CPU)
            encoder = variant.ffmpeg_encoder or variant.codec or variant.cpu_codec
            self.outputs_tree.insert(
                "",
                tk.END,
                iid=variant.id,
                values=(
                    "有効" if variant.enabled else "無効",
                    variant.name,
                    backend,
                    encoder,
                    resolution,
                    variant.folder_name,
                    variant.container,
                ),
            )

    def on_output_select(self, _event: object = None) -> None:
        selection = self.outputs_tree.selection()
        if not selection:
            return
        self.selected_output_id = selection[0]
        profile = self.current_profile()
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
                self.output_filename_template_var.set(variant.filename_template or "{source}")
                self.output_backend_var.set(variant.backend or (BACKEND_NVENC if variant.use_gpu else BACKEND_CPU))
                self.output_encoder_var.set(variant.ffmpeg_encoder or variant.codec or variant.cpu_codec)
                self.output_split_encode_mode_var.set(variant.split_encode_mode or "auto")
                self.output_gpu_choice_var.set(self._choice_for_variant_gpu(profile, variant))
                self.output_codec_var.set(variant.codec or profile.codec)
                self.output_cpu_codec_var.set(variant.cpu_codec or profile.cpu_codec)
                self.output_preset_var.set(variant.preset or profile.preset)
                self.output_cpu_preset_var.set(variant.cpu_preset or profile.cpu_preset)
                self.output_cpu_tune_var.set(variant.cpu_tune or getattr(profile, "cpu_tune", "none"))
                self.output_tune_var.set(variant.tune or profile.tune)
                self.output_rate_mode_var.set(variant.rate_mode or profile.rate_mode)
                self.output_cq_var.set(str(variant.cq_value if variant.cq_value is not None else profile.cq_value))
                self.output_bitrate_var.set(variant.bitrate or profile.bitrate)
                self.output_maxrate_var.set(variant.maxrate or profile.maxrate)
                self.output_bufsize_var.set(variant.bufsize or profile.bufsize)
                self.output_pix_fmt_var.set(variant.pix_fmt or profile.pix_fmt)
                self.output_scale_flags_var.set(variant.scale_flags or profile.scale_flags)
                self.output_audio_codec_var.set(variant.audio_codec or "copy")
                self.output_audio_bitrate_var.set(variant.audio_bitrate)
                self.output_audio_container_var.set(variant.audio_container)
                self.output_extra_input_args_var.set(variant.extra_input_args)
                self.output_extra_video_args_var.set(variant.extra_video_args)
                self.output_extra_audio_args_var.set(variant.extra_audio_args)
                self.output_extra_output_args_var.set(variant.extra_output_args)
                self.output_extra_concat_args_var.set(variant.extra_concat_args)
                self.output_extra_mux_args_var.set(variant.extra_mux_args)
                self.update_output_cpu_tune_choices()
                self.update_output_rate_controls()
                self.update_resolution_controls()
                self.render_output_resource_controls(profile, self.output_backend_var.get(), variant.resource_ids)
                self.update_output_encoder_controls()
                return

    def clear_output_selection(self) -> None:
        self.outputs_tree.selection_remove(self.outputs_tree.selection())
        self.selected_output_id = None
        self.set_output_edit_defaults(self.current_profile())

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

        rate_mode = self.output_rate_mode_var.get().upper()
        cq_value = self._cq_value_for_rate_mode(rate_mode, self.output_cq_var.get(), self._output_cq_fallback())
        if cq_value is None:
            messagebox.showerror("入力エラー", "CQ/CRF は数値で入力してください。")
            return
        backend = self.output_backend_var.get() or BACKEND_CPU
        if not backend_accepts_rate_mode(backend, rate_mode):
            messagebox.showerror("入力エラー", "CQ は CPU/NVENC のみで使用できます。QSV/AMF では VBR/ABR/CBR を選択してください。")
            return
        encoder = self.output_encoder_var.get().strip()
        resource_ids = self.selected_output_resource_ids()
        if not encoder:
            messagebox.showerror("入力エラー", "Encoder を選択してください。")
            return
        if not resource_ids:
            messagebox.showerror("入力エラー", "このエンコードセットで使うリソースを1つ以上選択してください。")
            return
        if any(resource_backend(resource_id) != backend for resource_id in resource_ids):
            messagebox.showerror("入力エラー", "1つのエンコードセット内で異なるbackendのリソースは混在できません。")
            return
        use_gpu = backend != BACKEND_CPU
        gpu_index = resource_index(resource_ids[0]) if use_gpu else 0
        gpu_name = ""
        if use_gpu:
            for resource in self.current_profile().hardware_resources:
                if resource.id == resource_ids[0]:
                    gpu_name = resource.label
                    break
        variant = OutputVariant(
            id=self.selected_output_id or new_id("variant"),
            name=name,
            folder_name=folder_name,
            height=height,
            container=container,
            enabled=self.output_enabled_var.get() if hasattr(self, "output_enabled_var") else True,
            filename_template=self.output_filename_template_var.get().strip() or "{source}",
            use_gpu=use_gpu,
            gpu_index=gpu_index,
            gpu_name=gpu_name,
            codec=encoder if backend == BACKEND_NVENC else self.output_codec_var.get(),
            cpu_codec=encoder if backend == BACKEND_CPU else self.output_cpu_codec_var.get(),
            backend=backend,
            ffmpeg_encoder=encoder,
            resource_ids=resource_ids,
            split_encode_mode=self.output_split_encode_mode_var.get(),
            preset=self.output_preset_var.get(),
            cpu_preset=self.output_cpu_preset_var.get(),
            cpu_tune=self.output_cpu_tune_var.get(),
            tune=self.output_tune_var.get(),
            rate_mode=rate_mode,
            cq_value=cq_value,
            bitrate=self.output_bitrate_var.get().strip(),
            maxrate=self.output_maxrate_var.get().strip(),
            bufsize=self.output_bufsize_var.get().strip(),
            pix_fmt=self.output_pix_fmt_var.get().strip(),
            scale_flags=self.output_scale_flags_var.get().strip(),
            audio_codec=self.output_audio_codec_var.get().strip(),
            audio_bitrate=self.output_audio_bitrate_var.get().strip(),
            audio_container=self.output_audio_container_var.get().strip(),
            extra_input_args=self.output_extra_input_args_var.get().strip(),
            extra_video_args=self.output_extra_video_args_var.get().strip(),
            extra_audio_args=self.output_extra_audio_args_var.get().strip(),
            extra_output_args=self.output_extra_output_args_var.get().strip(),
            extra_concat_args=self.output_extra_concat_args_var.get().strip(),
            extra_mux_args=self.output_extra_mux_args_var.get().strip(),
        )
        missing = missing_rate_fields(self.current_profile(), variant)
        if missing:
            messagebox.showerror("入力エラー", f"{variant.name}: {', '.join(missing)} を入力してください。")
            return

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
            resource_detail = self.resource_detail_label(job)
            message = job.message or f"{job.completed_segments}/{job.total_segments} segments"
            detail = f"{resource_detail} / {message}" if resource_detail else message
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
            self.refresh_encoder_capabilities(show_errors=False)
            self.log("FFmpeg / FFprobe is ready.")
        except Exception as exc:
            self.log(f"FFmpeg install failed: {exc}")
            self.root.after(0, lambda: messagebox.showerror("FFmpeg install failed", str(exc)))

    def refresh_encoder_capabilities(self, show_errors: bool = True) -> bool:
        try:
            self.encoder_capabilities = encoder_capabilities(self.paths.ffmpeg_path)
        except Exception as exc:
            self.encoder_capabilities = {}
            self.log(f"FFmpeg capability scan failed: {exc}")
            if show_errors:
                messagebox.showerror("FFmpeg error", f"FFmpeg encoder capability scan failed.\n{exc}")
            return False

        count = len(self.encoder_capabilities)
        if count:
            self.log(f"FFmpeg encoder capabilities loaded: {count} encoder(s).")
        else:
            self.log("FFmpeg encoder capability scan returned no encoders.")
        if hasattr(self, "root"):
            self.root.after(0, self.update_output_encoder_controls)
        return True

    def ensure_ffmpeg_before_run(self) -> bool:
        if self.paths.ffmpeg_path.exists() and self.paths.ffprobe_path.exists():
            try:
                ensure_ffmpeg_available(self.paths, auto_download=False, progress=self.log)
                return self.refresh_encoder_capabilities()
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
            return self.refresh_encoder_capabilities()
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
            messagebox.showerror("入力エラー", "出力プロファイルを1つ以上有効にしてください。")
            return False
        for variant in profile.outputs:
            if not variant.enabled:
                continue
            resources = variant_resource_ids(profile, variant)
            if not resources:
                messagebox.showerror("入力エラー", f"{variant.name}: 使用可能なリソースを選択してください。")
                return False
            expected_backend = variant.backend or resource_backend(resources[0])
            if any(resource_backend(resource_id) != expected_backend for resource_id in resources):
                messagebox.showerror("入力エラー", f"{variant.name}: backendの異なるリソースは混在できません。")
                return False
            rate_mode = (variant.rate_mode or profile.rate_mode or "CQ").upper()
            if not backend_accepts_rate_mode(expected_backend, rate_mode):
                messagebox.showerror("入力エラー", f"{variant.name}: CQ は CPU/NVENC のみで使用できます。QSV/AMF では VBR/ABR/CBR を選択してください。")
                return False
            missing = missing_rate_fields(profile, variant)
            if missing:
                labels = ", ".join(missing)
                messagebox.showerror("入力エラー", f"{variant.name}: {labels} を入力してください。")
                return False
        duplicates = duplicate_output_targets(profile)
        if duplicates:
            messagebox.showerror("入力エラー", f"同じ出力先が重複しています: {', '.join(duplicates)}")
            return False
        return True

    def validate_encoder_capabilities_before_run(self, profile: EncodeProfile, specs: List[JobSpec]) -> bool:
        if not self.encoder_capabilities and not self.refresh_encoder_capabilities():
            return False
        if not self.encoder_capabilities:
            messagebox.showerror("FFmpeg error", "FFmpeg encoder list could not be read.")
            return False

        smoke_targets: Dict[tuple[str, str, str], str] = {}
        for spec in specs:
            try:
                variant = variant_by_id(profile, spec.variant_id)
            except ValueError:
                continue
            encoder = encoder_codec(profile, variant)
            if encoder not in self.encoder_capabilities:
                messagebox.showerror("FFmpeg error", f"{variant.name}: FFmpeg encoder is not available: {encoder}")
                return False

            resource_ids = variant_resource_ids(profile, variant)
            resource_id = spec.assigned_resource_id if spec.assigned_resource_id in resource_ids else (resource_ids[0] if resource_ids else "")
            split_mode = str(variant.split_encode_mode or "").strip().lower()
            split_mode_applies = encoder in {"hevc_nvenc", "av1_nvenc"}
            if not split_mode_applies:
                split_mode = ""
            caps = self.encoder_capabilities.get(encoder, {})
            supports_split = bool(caps.get("supports_split_encode_mode"))
            split_modes = self.split_encode_modes_for_encoder(encoder)
            if split_mode and split_mode not in {"auto", "default"}:
                if not supports_split:
                    messagebox.showerror(
                        "FFmpeg error",
                        f"{variant.name}: {encoder} does not expose -split_encode_mode in this FFmpeg build.",
                    )
                    return False
                if split_modes and split_mode not in split_modes:
                    messagebox.showerror(
                        "FFmpeg error",
                        f"{variant.name}: split_encode_mode={split_mode} is not exposed by this FFmpeg build.",
                    )
                    return False

            smoke_targets[(encoder, resource_id, split_mode)] = variant.name

        for (encoder, resource_id, split_mode), variant_name in smoke_targets.items():
            self.log(f"Smoke test: {encoder} on {resource_id or 'default'}")
            ok, detail = smoke_test_encoder(
                self.paths.ffmpeg_path,
                encoder,
                resource_id=resource_id,
                split_encode_mode=split_mode,
                timeout=30,
            )
            if not ok:
                messagebox.showerror(
                    "FFmpeg error",
                    f"{variant_name}: encoder smoke test failed for {encoder} on {resource_id or 'default'}.\n{detail}",
                )
                self.log(f"Smoke test failed: {encoder} / {resource_id or 'default'} / {detail}")
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
        if not self.validate_encoder_capabilities_before_run(profile, specs):
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
        if not self.validate_encoder_capabilities_before_run(profile, specs):
            return

        self.start_specs(profile, specs, save=True)

    def assign_resources_to_specs(self, profile: EncodeProfile, specs: List[JobSpec]) -> List[JobSpec]:
        for spec in specs:
            try:
                variant = variant_by_id(profile, spec.variant_id)
            except ValueError:
                continue
            resource_ids = variant_resource_ids(profile, variant)
            if spec.assigned_resource_id not in resource_ids:
                spec.assigned_resource_id = resource_ids[0] if resource_ids else CPU_RESOURCE_ID
        return specs

    def start_specs(self, profile: EncodeProfile, specs: List[JobSpec], save: bool) -> None:
        specs = self.assign_resources_to_specs(profile, specs)
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
            self.active_resource_slots.clear()
            self.active_resource_slot_indexes.clear()
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
        resource_ids = variant_resource_ids(profile, variant)
        resource_id = spec.assigned_resource_id if spec.assigned_resource_id in resource_ids else (resource_ids[0] if resource_ids else "")
        spec.assigned_resource_id = resource_id
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
            resource_id=resource_id,
        )

    def resource_slot_limits(self, profile: EncodeProfile) -> Dict[str, int]:
        limits = {
            resource.id: max(1, int(resource.concurrency_slots))
            for resource in profile.hardware_resources
        }
        for variant in profile.outputs:
            for resource_id in variant_resource_ids(profile, variant):
                limits.setdefault(resource_id, 1)
        limits.setdefault(CPU_RESOURCE_ID, 1)
        return limits

    def resource_slot_weight(self, job: RuntimeJob, slot_limits: Dict[str, int]) -> int:
        limit = max(1, slot_limits.get(job.resource_id, 1))
        requested_segments = max(1, int(job.profile.max_parallel_jobs))
        return min(requested_segments, limit)

    def resource_detail_label(self, job: RuntimeJob) -> str:
        if not job.resource_id:
            return ""
        indexes = job.resource_slot_indexes or [job.resource_slot]
        labels = ",".join(str(index + 1) for index in indexes)
        return f"{job.resource_id} slots {labels}" if len(indexes) > 1 else f"{job.resource_id} slot {labels}"

    def reserve_resource_slots(self, job: RuntimeJob, slot_limits: Dict[str, int]) -> bool:
        resource_id = job.resource_id or CPU_RESOURCE_ID
        limit = max(1, slot_limits.get(resource_id, 1))
        weight = self.resource_slot_weight(job, slot_limits)
        used_indexes = self.active_resource_slot_indexes.setdefault(resource_id, set())
        available = [index for index in range(limit) if index not in used_indexes]
        if len(available) < weight:
            return False
        assigned_indexes = available[:weight]
        used_indexes.update(assigned_indexes)
        self.active_resource_slots[resource_id] = len(used_indexes)
        job.resource_id = resource_id
        job.resource_slot = assigned_indexes[0]
        job.resource_slot_indexes = assigned_indexes
        job.resource_slots_reserved = weight
        job.status = "準備中"
        self.active_jobs[job.job_id] = job
        return True

    def release_resource_slots(self, job: RuntimeJob) -> None:
        resource_id = job.resource_id or CPU_RESOURCE_ID
        reserved = max(1, int(job.resource_slots_reserved))
        with self.lock:
            used_indexes = self.active_resource_slot_indexes.get(resource_id, set())
            release_indexes = job.resource_slot_indexes or list(range(job.resource_slot, job.resource_slot + reserved))
            for index in release_indexes:
                used_indexes.discard(index)
            if used_indexes:
                self.active_resource_slot_indexes[resource_id] = used_indexes
                self.active_resource_slots[resource_id] = len(used_indexes)
            else:
                self.active_resource_slot_indexes.pop(resource_id, None)
                self.active_resource_slots.pop(resource_id, None)
            self.active_jobs.pop(job.job_id, None)

    def dequeue_runnable_jobs(self, profile: EncodeProfile) -> List[RuntimeJob]:
        slot_limits = self.resource_slot_limits(profile)
        runnable: List[RuntimeJob] = []
        deferred: List[RuntimeJob] = []
        with self.lock:
            while not self.pending_jobs.empty():
                try:
                    job = self.pending_jobs.get_nowait()
                except queue.Empty:
                    break
                if self.reserve_resource_slots(job, slot_limits):
                    runnable.append(job)
                else:
                    deferred.append(job)
            for job in deferred:
                self.pending_jobs.put(job)
        return runnable

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
                for job in self.dequeue_runnable_jobs(profile):
                    self.start_job_thread(job)

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
        joined_video = joined_video_path_for(self.paths, src, job.profile, job.variant)
        audio_out = temp_audio_path_for(self.paths, src, job.profile, job.variant)
        resource_detail = self.resource_detail_label(job) or "default"
        self.log(f"Start {job.variant.name}: {src.name} / {resource_detail}")
        self.log(f"Log file: {job.log_file}")

        segment_root.mkdir(parents=True, exist_ok=True)

        try:
            with open(job.log_file, "w", encoding="utf-8", errors="replace") as log_fp:
                duration = probe_duration(self.paths.ffprobe_path, src)
                ranges = segment_ranges(duration, segment_seconds)

                with self.lock:
                    job.total_segments = len(ranges)
                    job.message = f"0/{len(ranges)} segments"
                    job.segment_progress = {}
                    job.completed_segment_indexes = set()
                    job.status = "実行中"

                segment_files: List[Path] = [
                    segment_root / segment_file_name(job.variant, index, src=src)
                    for index, _range in enumerate(ranges)
                ]
                if not self.encode_segments_parallel(job, src, ranges, segment_files, segment_root, log_fp):
                    return
                if not self.wait_until_unpaused(job):
                    self._mark_job_cancelled(job)
                    return

                concat_file = segment_root / "concat.txt"
                write_concat_file(concat_file, segment_files)
                if joined_video.exists():
                    joined_video.unlink(missing_ok=True)
                concat_command = build_concat_command(
                    self.paths.ffmpeg_path,
                    concat_file,
                    joined_video,
                    job.profile,
                    job.variant,
                )
                log_fp.write("\nConcat command:\n")
                log_fp.write(command_to_text(concat_command) + "\n\n")
                log_fp.flush()

                with self.lock:
                    job.status = "結合中"
                    job.message = "finalizing"

                ret = self.run_process(job, concat_command, log_fp, None)
                if self.was_stopped() or ret != 0 or not joined_video.exists():
                    joined_video.unlink(missing_ok=True)
                    if self.was_stopped():
                        self._mark_job_cancelled(job)
                    else:
                        self._mark_job_failed(job, f"concat failed: exit {ret}")
                    return

                ffprobe_available = self.paths.ffprobe_path.exists()
                has_audio = probe_has_audio(self.paths.ffprobe_path, src)
                if has_audio:
                    if not self.wait_until_unpaused(job):
                        self._mark_job_cancelled(job)
                        return
                    if not audio_out.exists():
                        audio_command = build_audio_command(self.paths.ffmpeg_path, src, audio_out, job.profile, job.variant)
                        log_fp.write("\nAudio command:\n")
                        log_fp.write(command_to_text(audio_command) + "\n\n")
                        log_fp.flush()

                        with self.lock:
                            job.status = "音声処理中"
                            job.message = "processing audio"

                        ret = self.run_process(job, audio_command, log_fp, None)
                        if self.was_stopped() or ret != 0 or not audio_out.exists():
                            audio_out.unlink(missing_ok=True)
                            if self.was_stopped():
                                self._mark_job_cancelled(job)
                            else:
                                self._mark_job_failed(job, f"audio failed: exit {ret}")
                            return

                    if job.tmp_out.exists():
                        job.tmp_out.unlink(missing_ok=True)
                    mux_command = build_mux_command(
                        self.paths.ffmpeg_path,
                        joined_video,
                        audio_out,
                        job.tmp_out,
                        job.profile,
                        job.variant,
                    )
                    log_fp.write("\nMux command:\n")
                    log_fp.write(command_to_text(mux_command) + "\n\n")
                    log_fp.flush()

                    with self.lock:
                        job.status = "Mux中"
                        job.message = "muxing"

                    ret = self.run_process(job, mux_command, log_fp, None)
                    if self.was_stopped() or ret != 0 or not job.tmp_out.exists():
                        job.tmp_out.unlink(missing_ok=True)
                        if self.was_stopped():
                            self._mark_job_cancelled(job)
                        else:
                            self._mark_job_failed(job, f"mux failed: exit {ret}")
                        return
                else:
                    reason = "No audio stream detected" if ffprobe_available else "FFprobe unavailable"
                    self.skip_audio_output(job, src, joined_video, log_fp, reason)

                job.out_file.parent.mkdir(parents=True, exist_ok=True)
                if job.out_file.exists():
                    job.out_file.unlink()
                shutil.move(str(job.tmp_out), str(job.out_file))
                joined_video.unlink(missing_ok=True)
                audio_out.unlink(missing_ok=True)
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
            self.release_resource_slots(job)

    def encode_segments_parallel(
        self,
        job: RuntimeJob,
        src: Path,
        ranges: List[tuple[float, Optional[float]]],
        segment_files: List[Path],
        segment_root: Path,
        log_fp,
    ) -> bool:
        pending: queue.Queue[int] = queue.Queue()
        for index, final_segment in enumerate(segment_files):
            if final_segment.exists():
                self._mark_segment_done(job, index)
                continue
            pending.put(index)

        if pending.empty():
            return True

        failed = threading.Event()
        failure_message = {"text": ""}
        log_lock = threading.Lock()
        worker_count = min(
            max(1, int(job.profile.max_parallel_jobs)),
            max(1, int(job.resource_slots_reserved)),
            pending.qsize(),
        )

        def fail(message: str) -> None:
            if not failed.is_set():
                failure_message["text"] = message
                failed.set()
                with self.lock:
                    self.paused = False
                self.kill_job_processes(job)

        def worker() -> None:
            while not failed.is_set() and not self.was_stopped():
                try:
                    index = pending.get_nowait()
                except queue.Empty:
                    return
                start, duration_seconds = ranges[index]
                final_segment = segment_files[index]
                partial_segment = segment_root / segment_file_name(job.variant, index, partial=True, src=src)
                try:
                    if not self.wait_until_unpaused(job):
                        fail("cancelled")
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
                        resource_id=job.resource_id,
                    )
                    with log_lock:
                        log_fp.write(f"\nSegment {index + 1}/{len(ranges)} command:\n")
                        log_fp.write(command_to_text(command) + "\n\n")
                        log_fp.flush()

                    with self.lock:
                        job.current_segment = index + 1
                        job.status = "実行中"
                        job.message = f"segment {index + 1}/{len(ranges)}"

                    ret = self.run_process(
                        job,
                        command,
                        log_fp,
                        duration_seconds,
                        segment_index=index,
                        log_lock=log_lock,
                    )
                    if self.was_stopped():
                        fail("cancelled")
                        return
                    if ret != 0 or not partial_segment.exists():
                        partial_segment.unlink(missing_ok=True)
                        fail(f"segment {index + 1} failed: exit {ret}")
                        return

                    partial_segment.replace(final_segment)
                    self._mark_segment_done(job, index)
                finally:
                    pending.task_done()

        threads = [threading.Thread(target=worker, daemon=True) for _item in range(worker_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if self.was_stopped() or failure_message["text"] == "cancelled":
            self._mark_job_cancelled(job)
            return False
        if failed.is_set():
            self._mark_job_failed(job, failure_message["text"] or "segment failed")
            return False
        return True

    def skip_audio_output(self, job: RuntimeJob, src: Path, joined_video: Path, log_fp, reason: str) -> None:
        message = f"{reason}; keeping video-only output."
        log_fp.write(f"\nAudio skipped: {message}\n\n")
        log_fp.flush()
        self.log(f"{src.name}: {message}")
        with self.lock:
            job.status = "映像のみ"
            job.message = "video-only"
        if job.tmp_out.exists():
            job.tmp_out.unlink(missing_ok=True)
        shutil.move(str(joined_video), str(job.tmp_out))

    def run_process(
        self,
        job: RuntimeJob,
        command: List[str],
        log_fp,
        segment_duration: Optional[float],
        segment_index: Optional[int] = None,
        log_lock: Optional[threading.Lock] = None,
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
        process_slot = segment_index if segment_index is not None else -1
        with self.lock:
            job.process = process
            job.processes[process_slot] = process
            stop_requested = self.stop_requested
        if stop_requested and process.poll() is None:
            process.kill()

        try:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    if line:
                        if log_lock is None:
                            log_fp.write(line + "\n")
                            log_fp.flush()
                        else:
                            with log_lock:
                                log_fp.write(line + "\n")
                                log_fp.flush()
                        self.log(f"job {job.job_id}: {line}")
                        self._update_job_progress_from_line(job, line, segment_duration, segment_index)

            return process.wait()
        finally:
            with self.lock:
                if job.process is process:
                    job.process = None
                if job.processes.get(process_slot) is process:
                    job.processes.pop(process_slot, None)

    def _update_job_progress_from_line(
        self,
        job: RuntimeJob,
        line: str,
        segment_duration: Optional[float],
        segment_index: Optional[int],
    ) -> None:
        if not segment_duration or segment_index is None:
            return
        current = parse_ffmpeg_time(line)
        if current is None:
            return
        current_ratio = min(max(current / segment_duration, 0.0), 1.0)
        with self.lock:
            if segment_index not in job.completed_segment_indexes:
                job.segment_progress[segment_index] = current_ratio
            total_progress = sum(max(0.0, min(1.0, value)) for value in job.segment_progress.values())
            job.progress = (total_progress / max(job.total_segments, 1)) * 100.0

    def _mark_segment_done(self, job: RuntimeJob, segment_index: int) -> None:
        with self.lock:
            job.completed_segment_indexes.add(segment_index)
            job.segment_progress[segment_index] = 1.0
            job.completed_segments = len(job.completed_segment_indexes)
            total_progress = sum(max(0.0, min(1.0, value)) for value in job.segment_progress.values())
            job.progress = (total_progress / max(job.total_segments, 1)) * 100.0
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

    def kill_job_processes(self, job: RuntimeJob) -> None:
        with self.lock:
            processes = list(job.processes.values())
            if job.process is not None:
                processes.append(job.process)
        for process in processes:
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass

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
            self.kill_job_processes(job)
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
