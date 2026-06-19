from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Dict, List, Optional

try:
    from ffmpeg_nvenc_gui.core import (
        AUDIO_CODEC_CHOICES,
        BACKEND_AMF,
        BACKEND_CPU,
        BACKEND_NVENC,
        BACKEND_QSV,
        CPU_RESOURCE_ID,
        DEFAULT_ENCODER_BY_BACKEND,
        RESOLUTION_PRESETS,
        AppPaths,
        EncodeProfile,
        FileStatus,
        GpuInfo,
        HardwareResource,
        JobSpec,
        OutputVariant,
        audio_containers_for_codec,
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
        joined_video_path_for,
        load_profiles,
        load_state,
        missing_profile_dirs,
        missing_rate_fields,
        new_id,
        normalize_audio_codec,
        normalize_audio_container_for_codec,
        normalize_container_extension,
        normalize_profile_gpu,
        output_path_for,
        parse_ffmpeg_time,
        probe_duration,
        probe_has_audio,
        profile_archive_dir,
        profile_from_state,
        profile_input_dir,
        resource_backend,
        resource_id_for_backend,
        resource_index,
        resumable_specs,
        safe_folder_name,
        save_profiles,
        save_state,
        scan_profile_files,
        segment_dir_for,
        segment_file_name,
        segment_ranges,
        temp_audio_path_for,
        temp_output_path_for,
        variant_by_id,
        variant_resource_ids,
        variant_segment_minutes,
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
        AUDIO_CODEC_CHOICES,
        BACKEND_AMF,
        BACKEND_CPU,
        BACKEND_NVENC,
        BACKEND_QSV,
        CPU_RESOURCE_ID,
        DEFAULT_ENCODER_BY_BACKEND,
        RESOLUTION_PRESETS,
        AppPaths,
        EncodeProfile,
        FileStatus,
        GpuInfo,
        HardwareResource,
        JobSpec,
        OutputVariant,
        audio_containers_for_codec,
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
        joined_video_path_for,
        load_profiles,
        load_state,
        missing_profile_dirs,
        missing_rate_fields,
        new_id,
        normalize_audio_codec,
        normalize_audio_container_for_codec,
        normalize_container_extension,
        normalize_profile_gpu,
        output_path_for,
        parse_ffmpeg_time,
        probe_duration,
        probe_has_audio,
        profile_archive_dir,
        profile_from_state,
        profile_input_dir,
        resource_backend,
        resource_id_for_backend,
        resource_index,
        resumable_specs,
        safe_folder_name,
        save_profiles,
        save_state,
        scan_profile_files,
        segment_dir_for,
        segment_file_name,
        segment_ranges,
        temp_audio_path_for,
        temp_output_path_for,
        variant_by_id,
        variant_resource_ids,
        variant_segment_minutes,
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
CPU_PRESETS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
]
GPU_TUNES = ["none", "hq", "ll", "ull", "lossless"]
CPU_TUNES_BY_CODEC = {
    "libx264": ["none", "film", "animation", "grain", "stillimage", "fastdecode", "zerolatency", "psnr", "ssim"],
    "libx265": ["none", "psnr", "ssim", "grain", "fastdecode", "zerolatency"],
}
RATE_MODES = ["CQ", "VBR", "ABR", "CBR"]
CONTAINER_CHOICES = ["mp4", "mkv", "mov", "m4v", "webm", "ts", "m2ts"]
BACKEND_DISPLAY_NAMES = {
    BACKEND_CPU: "CPU",
    BACKEND_NVENC: "NVIDIA NVENC",
    BACKEND_QSV: "Intel QSV",
    BACKEND_AMF: "AMD AMF",
}
BRAND_COLORS = {
    BACKEND_CPU: ("CPU", "#6d5dfc", "#ffffff"),
    BACKEND_NVENC: ("NVIDIA", "#76b900", "#102000"),
    BACKEND_QSV: ("Intel", "#0071c5", "#ffffff"),
    BACKEND_AMF: ("AMD", "#ed1c24", "#ffffff"),
}
WORKSPACE_STACK_WIDTH = 1040
RUN_SUMMARY_STACK_WIDTH = 720
RUN_CONTROLS_STACK_WIDTH = 520


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


def default_output_backend_for_resource_ids(resource_ids: List[str]) -> str:
    for resource_id in resource_ids:
        backend = resource_backend(resource_id)
        if backend != BACKEND_CPU:
            return backend
    return BACKEND_CPU


def profile_uses_nvenc_resource(resource_ids: List[str]) -> bool:
    return any(resource_backend(resource_id) == BACKEND_NVENC for resource_id in resource_ids)


class SearchableCombobox(ttk.Combobox):
    def __init__(self, master: tk.Widget, values: List[str], **kwargs: object) -> None:
        self._search_values = list(values)
        super().__init__(master, values=self._search_values, **kwargs)
        self.bind("<KeyRelease>", self._filter_values, add="+")

    def set_values(self, values: List[str]) -> None:
        self._search_values = list(values)
        self.configure(values=self._search_values)

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


class OutputEditorSession:
    def __init__(self, app: "EncoderApp", profile: EncodeProfile, variant: Optional[OutputVariant]) -> None:
        self.app = app
        self.profile_id = profile.id
        self.variant_id = variant.id if variant is not None else None
        self.session_id = new_id("output_editor")
        self.dialog: Optional[tk.Toplevel] = None
        self.advanced_window: Optional[tk.Toplevel] = None
        self.resource_window: Optional[tk.Toplevel] = None
        self.resource_vars: Dict[str, tk.BooleanVar] = {}

        self.name_var = tk.StringVar(master=app.root)
        self.folder_var = tk.StringVar(master=app.root)
        self.resolution_var = tk.StringVar(master=app.root)
        self.custom_height_var = tk.StringVar(master=app.root)
        self.container_var = tk.StringVar(master=app.root)
        self.enabled_var = tk.BooleanVar(master=app.root)
        self.filename_template_var = tk.StringVar(master=app.root)
        self.input_dir_var = tk.StringVar(master=app.root)
        self.output_dir_var = tk.StringVar(master=app.root)
        self.segment_minutes_var = tk.StringVar(master=app.root)
        self.backend_var = tk.StringVar(master=app.root)
        self.encoder_var = tk.StringVar(master=app.root)
        self.split_encode_mode_var = tk.StringVar(master=app.root)
        self.codec_var = tk.StringVar(master=app.root)
        self.cpu_codec_var = tk.StringVar(master=app.root)
        self.preset_var = tk.StringVar(master=app.root)
        self.cpu_preset_var = tk.StringVar(master=app.root)
        self.cpu_tune_var = tk.StringVar(master=app.root)
        self.tune_var = tk.StringVar(master=app.root)
        self.rate_mode_var = tk.StringVar(master=app.root)
        self.cq_var = tk.StringVar(master=app.root)
        self.bitrate_var = tk.StringVar(master=app.root)
        self.maxrate_var = tk.StringVar(master=app.root)
        self.bufsize_var = tk.StringVar(master=app.root)
        self.pix_fmt_var = tk.StringVar(master=app.root)
        self.scale_flags_var = tk.StringVar(master=app.root)
        self.audio_codec_var = tk.StringVar(master=app.root)
        self.audio_bitrate_var = tk.StringVar(master=app.root)
        self.audio_container_var = tk.StringVar(master=app.root)
        self.extra_input_args_var = tk.StringVar(master=app.root)
        self.extra_video_args_var = tk.StringVar(master=app.root)
        self.extra_audio_args_var = tk.StringVar(master=app.root)
        self.extra_output_args_var = tk.StringVar(master=app.root)
        self.extra_concat_args_var = tk.StringVar(master=app.root)
        self.extra_mux_args_var = tk.StringVar(master=app.root)

        if variant is None:
            self.load_defaults(profile)
        else:
            self.load_variant(profile, variant)
        self.initial_snapshot = self.snapshot()

    def profile(self) -> Optional[EncodeProfile]:
        index = self.app.profile_index_by_id(self.profile_id)
        return self.app.profiles[index] if index is not None else None

    def load_defaults(self, profile: EncodeProfile) -> None:
        default_backend = default_output_backend_for_resource_ids(profile.resource_ids)
        default_encoder = DEFAULT_ENCODER_BY_BACKEND.get(default_backend, DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU])
        self.name_var.set("")
        self.folder_var.set("")
        self.resolution_var.set("1080p")
        self.custom_height_var.set("")
        self.container_var.set("mp4")
        self.enabled_var.set(True)
        self.filename_template_var.set("{source}")
        self.input_dir_var.set("")
        self.output_dir_var.set("")
        self.segment_minutes_var.set("")
        self.backend_var.set(default_backend)
        self.encoder_var.set(default_encoder)
        self.split_encode_mode_var.set("auto")
        self.codec_var.set(profile.codec)
        self.cpu_codec_var.set(profile.cpu_codec)
        self.preset_var.set(profile.preset)
        self.cpu_preset_var.set(profile.cpu_preset)
        self.cpu_tune_var.set(getattr(profile, "cpu_tune", "none"))
        self.tune_var.set(profile.tune)
        self.rate_mode_var.set(profile.rate_mode)
        self.cq_var.set(str(profile.cq_value))
        self.bitrate_var.set(profile.bitrate)
        self.maxrate_var.set(profile.maxrate)
        self.bufsize_var.set(profile.bufsize)
        self.pix_fmt_var.set(profile.pix_fmt)
        self.scale_flags_var.set(profile.scale_flags)
        self.audio_codec_var.set("copy")
        self.audio_bitrate_var.set("")
        self.audio_container_var.set("mka")
        self.extra_input_args_var.set("")
        self.extra_video_args_var.set("")
        self.extra_audio_args_var.set("")
        self.extra_output_args_var.set("")
        self.extra_concat_args_var.set("")
        self.extra_mux_args_var.set("")
        selected = [rid for rid in profile.resource_ids if resource_backend(rid) == default_backend]
        self.set_resource_selection(profile, default_backend, selected)

    def load_variant(self, profile: EncodeProfile, variant: OutputVariant) -> None:
        self.name_var.set(variant.name)
        self.folder_var.set(variant.folder_name)
        if variant.height is None:
            self.resolution_var.set("Original")
            self.custom_height_var.set("")
        elif variant.height in [2160, 1440, 1080, 720]:
            self.resolution_var.set(f"{variant.height}p")
            self.custom_height_var.set("")
        else:
            self.resolution_var.set("Custom")
            self.custom_height_var.set(str(variant.height))
        self.container_var.set(variant.container)
        self.enabled_var.set(variant.enabled)
        self.filename_template_var.set(variant.filename_template or "{source}")
        self.input_dir_var.set(variant.input_dir)
        self.output_dir_var.set(variant.output_dir)
        self.segment_minutes_var.set("" if variant.segment_minutes is None else str(variant.segment_minutes))
        backend = variant.backend or (BACKEND_NVENC if variant.use_gpu else BACKEND_CPU)
        self.backend_var.set(backend)
        self.encoder_var.set(variant.ffmpeg_encoder or variant.codec or variant.cpu_codec)
        self.split_encode_mode_var.set(variant.split_encode_mode or "auto")
        self.codec_var.set(variant.codec or profile.codec)
        self.cpu_codec_var.set(variant.cpu_codec or profile.cpu_codec)
        self.preset_var.set(variant.preset or profile.preset)
        self.cpu_preset_var.set(variant.cpu_preset or profile.cpu_preset)
        self.cpu_tune_var.set(variant.cpu_tune or getattr(profile, "cpu_tune", "none"))
        self.tune_var.set(variant.tune or profile.tune)
        self.rate_mode_var.set(variant.rate_mode or profile.rate_mode)
        self.cq_var.set(str(variant.cq_value if variant.cq_value is not None else profile.cq_value))
        self.bitrate_var.set(variant.bitrate or profile.bitrate)
        self.maxrate_var.set(variant.maxrate or profile.maxrate)
        self.bufsize_var.set(variant.bufsize or profile.bufsize)
        self.pix_fmt_var.set(variant.pix_fmt or profile.pix_fmt)
        self.scale_flags_var.set(variant.scale_flags or profile.scale_flags)
        self.audio_codec_var.set(normalize_audio_codec(variant.audio_codec or "copy"))
        self.audio_bitrate_var.set(variant.audio_bitrate)
        self.audio_container_var.set(
            normalize_audio_container_for_codec(self.audio_codec_var.get(), variant.audio_container)
        )
        self.extra_input_args_var.set(variant.extra_input_args)
        self.extra_video_args_var.set(variant.extra_video_args)
        self.extra_audio_args_var.set(variant.extra_audio_args)
        self.extra_output_args_var.set(variant.extra_output_args)
        self.extra_concat_args_var.set(variant.extra_concat_args)
        self.extra_mux_args_var.set(variant.extra_mux_args)
        self.set_resource_selection(profile, backend, list(variant.resource_ids))

    def set_resource_selection(self, profile: EncodeProfile, backend: str, selected: List[str]) -> None:
        compatible = [
            resource.id for resource in profile.hardware_resources if resource_backend(resource.id) == backend
        ]
        selected_ids = select_compatible_resource_ids(compatible, selected)
        self.resource_vars = {
            resource.id: tk.BooleanVar(master=self.app.root, value=resource.id in selected_ids)
            for resource in profile.hardware_resources
        }

    def selected_resource_ids(self) -> List[str]:
        return [resource_id for resource_id, var in self.resource_vars.items() if var.get()]

    def snapshot(self) -> tuple[object, ...]:
        return (
            self.name_var.get(),
            self.folder_var.get(),
            self.resolution_var.get(),
            self.custom_height_var.get(),
            self.container_var.get(),
            self.enabled_var.get(),
            self.filename_template_var.get(),
            self.input_dir_var.get(),
            self.output_dir_var.get(),
            self.segment_minutes_var.get(),
            self.backend_var.get(),
            self.encoder_var.get(),
            self.split_encode_mode_var.get(),
            self.codec_var.get(),
            self.cpu_codec_var.get(),
            self.preset_var.get(),
            self.cpu_preset_var.get(),
            self.cpu_tune_var.get(),
            self.tune_var.get(),
            self.rate_mode_var.get(),
            self.cq_var.get(),
            self.bitrate_var.get(),
            self.maxrate_var.get(),
            self.bufsize_var.get(),
            self.pix_fmt_var.get(),
            self.scale_flags_var.get(),
            self.audio_codec_var.get(),
            self.audio_bitrate_var.get(),
            self.audio_container_var.get(),
            self.extra_input_args_var.get(),
            self.extra_video_args_var.get(),
            self.extra_audio_args_var.get(),
            self.extra_output_args_var.get(),
            self.extra_concat_args_var.get(),
            self.extra_mux_args_var.get(),
            tuple(sorted(self.selected_resource_ids())),
        )

    def has_unsaved_changes(self) -> bool:
        return self.snapshot() != self.initial_snapshot

    def show(self) -> None:
        dialog = tk.Toplevel(self.app.root)
        self.dialog = dialog
        self.app.output_editor_sessions[self.session_id] = self
        dialog.title("出力プロファイル")
        dialog.transient(self.app.root)
        dialog.configure(bg=self.app.colors["bg"])
        dialog.geometry("860x560")
        dialog.minsize(720, 500)
        dialog.protocol("WM_DELETE_WINDOW", self.close)

        body = ttk.Frame(dialog, padding=14, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        form = self.app._dialog_surface(body, "出力プロファイル")
        form.pack(fill=tk.X)

        def grid_path(label: str, variable: tk.StringVar, row: int, pair: int = 0) -> None:
            label_column = pair * 2
            ttk.Label(form, text=label, style="Surface.TLabel").grid(
                row=row,
                column=label_column,
                sticky=tk.W,
                pady=4,
                padx=(0 if pair == 0 else 16, 8),
            )
            frame = ttk.Frame(form, style="Surface.TFrame")
            frame.grid(row=row, column=label_column + 1, sticky="ew", pady=4)
            frame.columnconfigure(0, weight=1)
            ttk.Entry(frame, textvariable=variable).grid(row=0, column=0, sticky="ew")
            ttk.Button(frame, text="参照", command=lambda: self.app.browse_dir(variable)).grid(
                row=0,
                column=1,
                padx=(6, 0),
            )

        self.app._dialog_entry(form, "名前", self.name_var, 1, 0)
        self.app._dialog_entry(form, "フォルダ名", self.folder_var, 1, 1)
        grid_path("入力先(空欄=基本)", self.input_dir_var, 2, 0)
        grid_path("出力先(空欄=基本)", self.output_dir_var, 2, 1)
        self.app._dialog_entry(form, "ファイル名テンプレート", self.filename_template_var, 3, 0)
        ttk.Checkbutton(form, text="このプロファイルを有効にする", variable=self.enabled_var).grid(
            row=3,
            column=2,
            columnspan=2,
            sticky=tk.W,
            pady=4,
            padx=(16, 0),
        )

        ttk.Label(form, text="エンコード先", style="Surface.TLabel").grid(
            row=4,
            column=0,
            sticky=tk.W,
            pady=4,
            padx=(0, 8),
        )
        self.resource_frame = ttk.Frame(form, padding=(8, 5), style="Inset.TFrame")
        self.resource_frame.grid(row=4, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Button(form, text="変更", command=self.open_resource_dialog).grid(row=4, column=3, sticky=tk.E, pady=4)

        self.resolution_combo = self.app._dialog_combo(
            form,
            "解像度",
            self.resolution_var,
            list(RESOLUTION_PRESETS.keys()),
            5,
            0,
        )
        self.resolution_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_resolution_controls())
        self.custom_height_label = ttk.Label(form, text="カスタム高さ", style="Surface.TLabel")
        self.custom_height_entry = ttk.Entry(form, textvariable=self.custom_height_var)
        self.custom_height_label.grid(row=5, column=2, sticky=tk.W, pady=4, padx=(16, 8))
        self.custom_height_entry.grid(row=5, column=3, sticky="ew", pady=4)

        self.container_combo = self.app._dialog_search_combo(
            form,
            "コンテナ",
            self.container_var,
            CONTAINER_CHOICES,
            6,
            0,
        )
        self.encoder_combo = self.app._dialog_combo(
            form,
            "Encoder",
            self.encoder_var,
            self.app._encoders_for_backend(self.backend_var.get()),
            6,
            1,
        )
        self.encoder_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_encoder_controls())
        self.rate_combo = self.app._dialog_combo(form, "Rate", self.rate_mode_var, RATE_MODES, 7, 0)
        self.rate_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_encoder_controls())
        self.cq_entry = self.app._dialog_entry(form, "CQ/CRF", self.cq_var, 7, 1)
        self.bitrate_entry = self.app._dialog_entry(form, "Bitrate", self.bitrate_var, 8, 0)

        bottom = ttk.Frame(body, style="App.TFrame")
        bottom.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(bottom, text="詳細設定", command=self.open_advanced_dialog).pack(side=tk.LEFT)
        ttk.Button(bottom, text="保存", style="Accent.TButton", command=self.save_and_close).pack(
            side=tk.RIGHT,
            padx=(8, 0),
        )
        ttk.Button(bottom, text="閉じる", command=self.close).pack(side=tk.RIGHT)

        self.render_resource_controls()
        self.update_resolution_controls()
        self.update_encoder_controls()
        self.initial_snapshot = self.snapshot()

    def render_resource_controls(self) -> None:
        if not hasattr(self, "resource_frame"):
            return
        for child in self.resource_frame.winfo_children():
            child.destroy()
        profile = self.profile()
        if profile is None:
            ttk.Label(self.resource_frame, text="プロファイルが見つかりません", style="Inset.TLabel").pack(side=tk.LEFT)
            return
        resources = list(profile.hardware_resources)
        if not resources:
            ttk.Label(self.resource_frame, text="利用できるリソースがありません", style="Inset.TLabel").pack(
                side=tk.LEFT
            )
            return
        backend = self.backend_var.get() or BACKEND_CPU
        selected = self.selected_resource_ids()
        compatible = [resource.id for resource in resources if resource_backend(resource.id) == backend]
        selected = select_compatible_resource_ids(compatible, selected)
        for resource_id, var in self.resource_vars.items():
            var.set(resource_id in selected)
        if selected:
            backend = resource_backend(selected[0])
            self.backend_var.set(backend)
            self.app._brand_badge(self.resource_frame, backend).pack(side=tk.LEFT, padx=(0, 8))
        labels = self.app.selected_resource_labels(profile, selected)
        ttk.Label(
            self.resource_frame,
            text=f"{self.app._backend_display_name(backend)} / {', '.join(labels)}",
            style="Inset.TLabel",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def open_resource_dialog(self) -> None:
        if self.app._widget_exists(self.resource_window):
            self.resource_window.lift()
            return
        profile = self.profile()
        if profile is None:
            messagebox.showerror("入力エラー", "プロファイルが見つかりません。", parent=self.dialog)
            return
        resources = list(profile.hardware_resources)
        if not resources:
            messagebox.showerror("入力エラー", "利用できるリソースがありません。", parent=self.dialog)
            return

        dialog = tk.Toplevel(self.dialog or self.app.root)
        self.resource_window = dialog
        dialog.title("エンコード先")
        dialog.transient(self.dialog or self.app.root)
        dialog.configure(bg=self.app.colors["bg"])
        dialog.resizable(False, False)

        def on_close() -> None:
            self.resource_window = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)
        body = ttk.Frame(dialog, padding=16, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="エンコード先", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 10))
        resource_vars = {
            resource.id: tk.BooleanVar(
                master=self.app.root,
                value=self.resource_vars[resource.id].get() if resource.id in self.resource_vars else False,
            )
            for resource in resources
        }

        def on_change(changed_id: str) -> None:
            if not resource_vars[changed_id].get():
                return
            backend = resource_backend(changed_id)
            for resource_id, var in resource_vars.items():
                if resource_id != changed_id and resource_backend(resource_id) != backend:
                    var.set(False)

        grouped: Dict[str, ttk.Frame] = {}
        for resource in resources:
            backend = resource_backend(resource.id)
            if backend not in grouped:
                group = ttk.Frame(body, padding=10, style="Surface.TFrame", relief=tk.RAISED, borderwidth=1)
                group.pack(fill=tk.X, pady=(0, 10))
                head = ttk.Frame(group, style="Surface.TFrame")
                head.pack(fill=tk.X, pady=(0, 6))
                self.app._brand_badge(head, backend).pack(side=tk.LEFT, padx=(0, 8))
                ttk.Label(head, text=self.app._backend_display_name(backend), style="Surface.TLabel").pack(side=tk.LEFT)
                grouped[backend] = group
            row = ttk.Frame(grouped[backend], style="Surface.TFrame")
            row.pack(fill=tk.X, pady=2)
            ttk.Checkbutton(
                row,
                text=resource.label,
                variable=resource_vars[resource.id],
                command=lambda resource_id=resource.id: on_change(resource_id),
            ).pack(side=tk.LEFT)
            detail = f"{resource.concurrency_slots} slot(s)"
            if resource.detected_encoder_engines:
                detail = f"NVENC {resource.detected_encoder_engines} engine(s)"
            elif resource.detection_error:
                detail = "NVENCエンジン数未検出、slotsは手動調整"
            ttk.Label(row, text=detail, style="Muted.TLabel").pack(side=tk.RIGHT)

        buttons = ttk.Frame(body, style="App.TFrame")
        buttons.pack(fill=tk.X, pady=(4, 0))

        def apply_selection() -> None:
            selected_ids = [resource_id for resource_id, var in resource_vars.items() if var.get()]
            if not selected_ids:
                messagebox.showerror("入力エラー", "リソースを1つ以上選択してください。", parent=dialog)
                return
            backend = resource_backend(selected_ids[0])
            if any(resource_backend(resource_id) != backend for resource_id in selected_ids):
                messagebox.showerror(
                    "入力エラー", "1つの出力で異なるbackendのリソースは混在できません。", parent=dialog
                )
                return
            self.resource_vars = {
                resource.id: tk.BooleanVar(master=self.app.root, value=resource.id in selected_ids)
                for resource in resources
            }
            self.backend_var.set(backend)
            self.update_encoder_controls()
            on_close()

        ttk.Button(buttons, text="適用", style="Accent.TButton", command=apply_selection).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="閉じる", command=on_close).pack(side=tk.RIGHT, padx=(0, 8))

    def open_advanced_dialog(self) -> None:
        if self.app._widget_exists(self.advanced_window):
            self.advanced_window.lift()
            return
        dialog = tk.Toplevel(self.dialog or self.app.root)
        self.advanced_window = dialog
        dialog.title("出力の詳細設定")
        dialog.transient(self.dialog or self.app.root)
        dialog.configure(bg=self.app.colors["bg"])
        dialog.geometry("720x560")

        def on_close() -> None:
            self.advanced_window = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)
        body = ttk.Frame(dialog, padding=14, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        tabs = ttk.Notebook(body, style="Dialog.TNotebook")
        tabs.pack(fill=tk.BOTH, expand=True)
        video_tab = ttk.Frame(tabs, padding=10, style="App.TFrame")
        audio_tab = ttk.Frame(tabs, padding=10, style="App.TFrame")
        ffmpeg_tab = ttk.Frame(tabs, padding=10, style="App.TFrame")
        tabs.add(video_tab, text="映像")
        tabs.add(audio_tab, text="音声")
        tabs.add(ffmpeg_tab, text="FFmpeg")

        self.engine_panel_container = ttk.Frame(video_tab, style="App.TFrame")
        self.engine_panel_container.pack(fill=tk.X)
        self.nvenc_detail_frame = self.app._dialog_surface(self.engine_panel_container, "NVENC")
        self.preset_combo = self.app._dialog_combo(
            self.nvenc_detail_frame,
            "NVENC Preset",
            self.preset_var,
            GPU_PRESETS,
            1,
            0,
        )
        self.tune_combo = self.app._dialog_combo(
            self.nvenc_detail_frame,
            "NVENC Tune",
            self.tune_var,
            GPU_TUNES,
            1,
            1,
        )
        self.split_combo = self.app._dialog_combo(
            self.nvenc_detail_frame,
            "SFE",
            self.split_encode_mode_var,
            ["auto"],
            2,
            0,
            width=18,
        )
        self.cpu_detail_frame = self.app._dialog_surface(self.engine_panel_container, "CPU")
        self.cpu_preset_combo = self.app._dialog_combo(
            self.cpu_detail_frame,
            "CPU Preset",
            self.cpu_preset_var,
            CPU_PRESETS,
            1,
            0,
        )
        self.cpu_tune_combo = self.app._dialog_combo(
            self.cpu_detail_frame,
            "CPU Tune",
            self.cpu_tune_var,
            CPU_TUNES_BY_CODEC.get(self.cpu_codec_var.get(), CPU_TUNES_BY_CODEC["libx264"]),
            1,
            1,
        )
        self.hardware_detail_frame = self.app._dialog_surface(self.engine_panel_container, "Hardware")
        ttk.Label(
            self.hardware_detail_frame,
            text="QSV/AMF固有の調整はFFmpeg引数タブで指定できます。",
            style="Surface.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W)

        rate = self.app._dialog_surface(video_tab, "レート制御")
        rate.pack(fill=tk.X, pady=(10, 0))
        self.maxrate_entry = self.app._dialog_entry(rate, "Maxrate", self.maxrate_var, 1, 0)
        self.bufsize_entry = self.app._dialog_entry(rate, "Bufsize", self.bufsize_var, 1, 1)

        format_box = self.app._dialog_surface(video_tab, "フォーマット")
        format_box.pack(fill=tk.X, pady=(10, 0))
        self.app._dialog_entry(format_box, "分割間隔(分・空欄=基本)", self.segment_minutes_var, 1, 0)
        self.app._dialog_entry(format_box, "Pix fmt", self.pix_fmt_var, 1, 1)
        self.app._dialog_entry(format_box, "Scale flags", self.scale_flags_var, 2, 0)

        audio = self.app._dialog_surface(audio_tab, "音声")
        audio.pack(fill=tk.X)
        self.audio_codec_combo = self.app._dialog_search_combo(
            audio,
            "Audio codec",
            self.audio_codec_var,
            AUDIO_CODEC_CHOICES,
            1,
            0,
        )
        self.audio_codec_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_audio_container_choices())
        self.app._dialog_entry(audio, "Audio bitrate", self.audio_bitrate_var, 1, 1)
        self.audio_container_combo = self.app._dialog_search_combo(
            audio,
            "Audio container",
            self.audio_container_var,
            audio_containers_for_codec(self.audio_codec_var.get()),
            2,
            0,
        )
        self.audio_container_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_audio_container_choices())

        ffmpeg = self.app._dialog_surface(ffmpeg_tab, "追加引数")
        ffmpeg.pack(fill=tk.X)
        self.app._dialog_entry(ffmpeg, "FFmpeg input args", self.extra_input_args_var, 1, 0)
        self.app._dialog_entry(ffmpeg, "FFmpeg video args", self.extra_video_args_var, 1, 1)
        self.app._dialog_entry(ffmpeg, "FFmpeg audio args", self.extra_audio_args_var, 2, 0)
        self.app._dialog_entry(ffmpeg, "FFmpeg output args", self.extra_output_args_var, 2, 1)
        self.app._dialog_entry(ffmpeg, "FFmpeg concat args", self.extra_concat_args_var, 3, 0)
        self.app._dialog_entry(ffmpeg, "FFmpeg mux args", self.extra_mux_args_var, 3, 1)

        bottom = ttk.Frame(body, style="App.TFrame")
        bottom.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(bottom, text="閉じる", style="Accent.TButton", command=on_close).pack(side=tk.RIGHT)
        self.update_cpu_tune_choices()
        self.update_audio_container_choices()
        self.update_encoder_controls()

    def _set_grid_pair_visible(self, widget: object, visible: bool) -> None:
        self.app._set_grid_pair_visible(widget, visible)

    def update_cpu_tune_choices(self) -> None:
        combo = getattr(self, "cpu_tune_combo", None)
        if not self.app._widget_exists(combo):
            return
        values = CPU_TUNES_BY_CODEC.get(self.cpu_codec_var.get(), CPU_TUNES_BY_CODEC["libx264"])
        combo.configure(values=values)
        if self.cpu_tune_var.get() not in values:
            self.cpu_tune_var.set("none")

    def update_audio_container_choices(self) -> None:
        codec = normalize_audio_codec(self.audio_codec_var.get())
        self.audio_codec_var.set(codec)
        values = audio_containers_for_codec(codec)
        combo = getattr(self, "audio_container_combo", None)
        if self.app._widget_exists(combo):
            if hasattr(combo, "set_values"):
                combo.set_values(values)
            else:
                combo.configure(values=values)
        self.audio_container_var.set(normalize_audio_container_for_codec(codec, self.audio_container_var.get()))

    def update_rate_controls(self) -> None:
        if not self.app._widget_exists(getattr(self, "cq_entry", None)):
            return
        mode = self.rate_mode_var.get().upper()
        backend = self.backend_var.get() or BACKEND_CPU
        controls = [
            (getattr(self, "cq_entry", None), backend_allows_cq(backend) and mode in {"CQ", "VBR"}),
            (getattr(self, "bitrate_entry", None), mode in {"VBR", "ABR", "CBR"}),
            (getattr(self, "maxrate_entry", None), mode == "VBR"),
            (getattr(self, "bufsize_entry", None), mode in {"VBR", "CBR"}),
        ]
        for widget, visible in controls:
            self._set_grid_pair_visible(widget, visible)

    def update_resolution_controls(self) -> None:
        if not self.app._widget_exists(getattr(self, "custom_height_entry", None)):
            return
        if self.resolution_var.get() == "Custom":
            self.custom_height_label.grid()
            self.custom_height_entry.grid()
            self.custom_height_entry.configure(state=tk.NORMAL)
        else:
            self.custom_height_var.set("")
            self.custom_height_label.grid_remove()
            self.custom_height_entry.grid_remove()

    def sync_advanced_panels(self, backend: str) -> None:
        panels = [
            getattr(self, "nvenc_detail_frame", None),
            getattr(self, "cpu_detail_frame", None),
            getattr(self, "hardware_detail_frame", None),
        ]
        for panel in panels:
            if self.app._widget_exists(panel):
                panel.pack_forget()
        if backend == BACKEND_CPU:
            panel = getattr(self, "cpu_detail_frame", None)
        elif backend == BACKEND_NVENC:
            panel = getattr(self, "nvenc_detail_frame", None)
        else:
            panel = getattr(self, "hardware_detail_frame", None)
        if self.app._widget_exists(panel):
            panel.pack(fill=tk.X)

    def update_encoder_controls(self) -> None:
        if not self.app._widget_exists(getattr(self, "encoder_combo", None)):
            return
        profile = self.profile()
        if profile is None:
            return
        backend = self.backend_var.get() or BACKEND_CPU
        self.render_resource_controls()
        selected = self.selected_resource_ids()
        backend = self.backend_var.get() or backend
        encoders = self.app._encoders_for_backend(backend, selected, verify_nvenc=False)
        self.encoder_combo.configure(values=encoders)
        if self.encoder_var.get() not in encoders:
            self.encoder_var.set(encoders[0] if encoders else "")

        encoder = self.encoder_var.get()
        if backend == BACKEND_NVENC:
            self.codec_var.set(encoder)
        elif backend == BACKEND_CPU:
            self.cpu_codec_var.set(encoder)
        self.update_cpu_tune_choices()
        rate_modes = rate_modes_for_backend(backend)
        if self.app._widget_exists(getattr(self, "rate_combo", None)):
            self.rate_combo.configure(values=rate_modes)
        if self.rate_mode_var.get().upper() not in rate_modes:
            self.rate_mode_var.set(rate_modes[0])
        self.update_rate_controls()
        self.sync_advanced_panels(backend)

        nvenc_state = tk.NORMAL if backend == BACKEND_NVENC else tk.DISABLED
        cpu_state = tk.NORMAL if backend == BACKEND_CPU else tk.DISABLED
        for widget in (getattr(self, "preset_combo", None), getattr(self, "tune_combo", None)):
            if self.app._widget_exists(widget):
                widget.configure(state=nvenc_state)
        for widget in (getattr(self, "cpu_preset_combo", None), getattr(self, "cpu_tune_combo", None)):
            if self.app._widget_exists(widget):
                widget.configure(state=cpu_state)

        split_modes = self.app.split_encode_modes_for_encoder(encoder)
        split_combo = getattr(self, "split_combo", None)
        if backend == BACKEND_NVENC and encoder in {"hevc_nvenc", "av1_nvenc"} and split_modes:
            if self.app._widget_exists(split_combo):
                split_combo.configure(values=split_modes, state=tk.NORMAL)
                self._set_grid_pair_visible(split_combo, True)
        else:
            split_modes = ["auto"]
            self.split_encode_mode_var.set("auto")
            if self.app._widget_exists(split_combo):
                split_combo.configure(values=["auto"], state=tk.NORMAL)
                self._set_grid_pair_visible(split_combo, False)
        if self.split_encode_mode_var.get() not in split_modes:
            self.split_encode_mode_var.set(split_modes[0])

    def build_variant(self) -> Optional[OutputVariant]:
        profile = self.profile()
        if profile is None:
            messagebox.showerror("入力エラー", "プロファイルが見つかりません。", parent=self.dialog)
            return None
        name = self.name_var.get().strip()
        folder_name = safe_folder_name(self.folder_var.get().strip() or name)
        container = normalize_container_extension(self.container_var.get(), default="")
        if not name:
            messagebox.showerror("入力エラー", "出力名を入力してください。", parent=self.dialog)
            return None
        if not container:
            messagebox.showerror("入力エラー", "コンテナは英数字で入力してください。例: mp4, mkv", parent=self.dialog)
            return None

        preset = self.resolution_var.get()
        if preset == "Custom":
            try:
                height = int(self.custom_height_var.get())
            except ValueError:
                messagebox.showerror("入力エラー", "カスタム解像度の高さを数値で入力してください。", parent=self.dialog)
                return None
            if height < 1:
                messagebox.showerror("入力エラー", "カスタム解像度は1以上にしてください。", parent=self.dialog)
                return None
        else:
            height = RESOLUTION_PRESETS.get(preset)

        raw_segment_minutes = self.segment_minutes_var.get().strip()
        segment_minutes: Optional[int] = None
        if raw_segment_minutes:
            try:
                segment_minutes = int(raw_segment_minutes)
            except ValueError:
                messagebox.showerror("入力エラー", "分割間隔は数値で入力してください。", parent=self.dialog)
                return None
            if segment_minutes < 1:
                messagebox.showerror("入力エラー", "分割間隔は1分以上にしてください。", parent=self.dialog)
                return None

        backend = self.backend_var.get() or BACKEND_CPU
        rate_mode = self.rate_mode_var.get().upper()
        if not backend_accepts_rate_mode(backend, rate_mode):
            messagebox.showerror(
                "入力エラー",
                "CQ は CPU/NVENC のみで使用できます。QSV/AMF では VBR/ABR/CBR を選択してください。",
                parent=self.dialog,
            )
            return None
        cq_value = (
            self.app._cq_value_for_rate_mode(rate_mode, self.cq_var.get(), profile.cq_value)
            if backend_allows_cq(backend)
            else profile.cq_value
        )
        if cq_value is None:
            messagebox.showerror("入力エラー", "CQ/CRF は数値で入力してください。", parent=self.dialog)
            return None
        encoder = self.encoder_var.get().strip()
        resource_ids = self.selected_resource_ids()
        if not encoder:
            messagebox.showerror("入力エラー", "Encoder を選択してください。", parent=self.dialog)
            return None
        if not resource_ids:
            messagebox.showerror(
                "入力エラー", "このエンコード設定で使うリソースを1つ以上選択してください。", parent=self.dialog
            )
            return None
        if any(resource_backend(resource_id) != backend for resource_id in resource_ids):
            messagebox.showerror(
                "入力エラー", "1つのエンコード設定内で異なるbackendのリソースは混在できません。", parent=self.dialog
            )
            return None

        use_gpu = backend != BACKEND_CPU
        gpu_index = resource_index(resource_ids[0]) if use_gpu else 0
        gpu_name = ""
        if use_gpu:
            for resource in profile.hardware_resources:
                if resource.id == resource_ids[0]:
                    gpu_name = resource.label
                    break
        audio_codec = normalize_audio_codec(self.audio_codec_var.get())
        audio_container = normalize_audio_container_for_codec(audio_codec, self.audio_container_var.get())
        variant = OutputVariant(
            id=self.variant_id or new_id("variant"),
            name=name,
            folder_name=folder_name,
            height=height,
            container=container,
            enabled=self.enabled_var.get(),
            filename_template=self.filename_template_var.get().strip() or "{source}",
            input_dir=self.input_dir_var.get().strip(),
            output_dir=self.output_dir_var.get().strip(),
            segment_minutes=segment_minutes,
            use_gpu=use_gpu,
            gpu_index=gpu_index,
            gpu_name=gpu_name,
            codec=encoder if backend == BACKEND_NVENC else self.codec_var.get(),
            cpu_codec=encoder if backend == BACKEND_CPU else self.cpu_codec_var.get(),
            backend=backend,
            ffmpeg_encoder=encoder,
            resource_ids=resource_ids,
            split_encode_mode=self.split_encode_mode_var.get(),
            preset=self.preset_var.get(),
            cpu_preset=self.cpu_preset_var.get(),
            cpu_tune=self.cpu_tune_var.get(),
            tune=self.tune_var.get(),
            rate_mode=rate_mode,
            cq_value=cq_value,
            bitrate=self.bitrate_var.get().strip(),
            maxrate=self.maxrate_var.get().strip(),
            bufsize=self.bufsize_var.get().strip(),
            pix_fmt=self.pix_fmt_var.get().strip(),
            scale_flags=self.scale_flags_var.get().strip(),
            audio_codec=audio_codec,
            audio_bitrate=self.audio_bitrate_var.get().strip(),
            audio_container=audio_container,
            extra_input_args=self.extra_input_args_var.get().strip(),
            extra_video_args=self.extra_video_args_var.get().strip(),
            extra_audio_args=self.extra_audio_args_var.get().strip(),
            extra_output_args=self.extra_output_args_var.get().strip(),
            extra_concat_args=self.extra_concat_args_var.get().strip(),
            extra_mux_args=self.extra_mux_args_var.get().strip(),
        )
        missing = missing_rate_fields(profile, variant)
        if missing:
            messagebox.showerror(
                "入力エラー", f"{variant.name}: {', '.join(missing)} を入力してください。", parent=self.dialog
            )
            return None
        return variant

    def target_outputs(self) -> List[OutputVariant]:
        profile = self.profile()
        if profile is None:
            return []
        source = self.app.editing_outputs if self.app.active_profile_id == self.profile_id else profile.outputs
        return [OutputVariant.from_dict(asdict(item)) for item in source]

    def save_and_close(self) -> bool:
        variant = self.build_variant()
        if variant is None:
            return False
        profile = self.profile()
        if profile is None:
            messagebox.showerror("入力エラー", "プロファイルが見つかりません。", parent=self.dialog)
            return False

        next_outputs: List[OutputVariant] = []
        replaced = False
        for existing in self.target_outputs():
            if existing.id == variant.id:
                next_outputs.append(variant)
                replaced = True
            else:
                next_outputs.append(existing)
        if not replaced:
            next_outputs.append(variant)
        duplicates = duplicate_output_targets_for_outputs(next_outputs)
        if duplicates:
            messagebox.showerror(
                "入力エラー", f"同じ出力先が重複しています: {', '.join(duplicates)}", parent=self.dialog
            )
            return False

        profile.outputs = [OutputVariant.from_dict(asdict(item)) for item in next_outputs]
        try:
            save_profiles(self.app.paths, self.app.profiles)
        except OSError as exc:
            messagebox.showerror("保存エラー", str(exc), parent=self.dialog)
            return False

        if self.app.active_profile_id == self.profile_id:
            self.app.editing_outputs = [OutputVariant.from_dict(asdict(item)) for item in profile.outputs]
            self.app.selected_output_id = variant.id
            self.app.refresh_outputs_tree()
            self.app.outputs_tree.selection_set(variant.id)
            self.app.on_output_select()
            self.app.scan_files()
        self.app.log(f"出力プロファイルを保存: {variant.name}")
        self.variant_id = variant.id
        self.initial_snapshot = self.snapshot()
        self.destroy()
        return True

    def close(self) -> None:
        if self.has_unsaved_changes():
            result = messagebox.askyesnocancel(
                "未保存の変更",
                "保存していない変更があります。保存して閉じますか？",
                parent=self.dialog,
            )
            if result is None:
                return
            if result:
                self.save_and_close()
                return
        self.destroy()

    def destroy(self) -> None:
        if self.app._widget_exists(self.advanced_window):
            self.advanced_window.destroy()
        if self.app._widget_exists(self.resource_window):
            self.resource_window.destroy()
        if self.app._widget_exists(self.dialog):
            self.dialog.destroy()
        self.app.output_editor_sessions.pop(self.session_id, None)


class EncoderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.ui_thread = threading.current_thread()
        self.root.title("NVEnc Archive Studio")
        self.root.geometry("1180x720")
        self.root.minsize(760, 560)

        self.paths: AppPaths = build_paths()
        ensure_dirs(self.paths)

        self.gpus: List[GpuInfo] = detect_nvidia_gpus()
        self.profiles: List[EncodeProfile] = load_profiles(self.paths, self.gpus)
        self.active_profile_id: Optional[str] = self.profiles[0].id if self.profiles else None
        self.files: List[FileStatus] = []
        self.editing_outputs: List[OutputVariant] = []
        self.selected_output_id: Optional[str] = None
        self.output_editor_sessions: Dict[str, object] = {}

        self.job_counter = 0
        self.pending_jobs: queue.Queue[RuntimeJob] = queue.Queue()
        self.active_jobs: Dict[int, RuntimeJob] = {}
        self.all_jobs: Dict[int, RuntimeJob] = {}
        self.job_rows: Dict[int, str] = {}
        self.active_resource_slots: Dict[str, int] = {}
        self.active_resource_slot_indexes: Dict[str, set[int]] = {}
        self.encoder_capabilities: Dict[str, Dict[str, object]] = {}
        self.encoder_smoke_cache: Dict[tuple[str, str, str], tuple[bool, str]] = {}

        self.lock = threading.Lock()
        self.log_queue: queue.Queue[str] = queue.Queue()

        self.running = False
        self.preparing = False
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
            "bg": "#e8edf5",
            "surface": "#f4f7fb",
            "surface_alt": "#edf2f8",
            "highlight": "#ffffff",
            "text": "#162033",
            "muted": "#607086",
            "accent": "#2563eb",
            "accent_hover": "#1d4ed8",
            "danger": "#dc2626",
            "log_bg": "#142033",
            "log_fg": "#d8f3dc",
        }

        self.root.configure(bg=self.colors["bg"])
        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure(
            "Surface.TFrame",
            background=self.colors["surface"],
            relief="flat",
            borderwidth=0,
        )
        style.configure("Inset.TFrame", background=self.colors["surface_alt"], relief="sunken", borderwidth=1)
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["text"])
        style.configure("Surface.TLabel", background=self.colors["surface"], foreground=self.colors["text"])
        style.configure("Muted.TLabel", background=self.colors["surface"], foreground=self.colors["muted"])
        style.configure("Inset.TLabel", background=self.colors["surface_alt"], foreground=self.colors["text"])
        style.configure(
            "Title.TLabel", background=self.colors["bg"], foreground=self.colors["text"], font=("Segoe UI", 20, "bold")
        )
        style.configure(
            "Subtitle.TLabel", background=self.colors["bg"], foreground=self.colors["muted"], font=("Segoe UI", 10)
        )
        style.configure(
            "Section.TLabel",
            background=self.colors["surface"],
            foreground=self.colors["text"],
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "Accent.TButton",
            background=self.colors["accent"],
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 8),
            relief="raised",
            borderwidth=1,
        )
        style.map(
            "Accent.TButton", background=[("active", self.colors["accent_hover"])], foreground=[("disabled", "#e5e7eb")]
        )
        style.configure(
            "Danger.TButton",
            foreground=self.colors["danger"],
            font=("Segoe UI", 10, "bold"),
            padding=(12, 7),
            relief="raised",
            borderwidth=1,
        )
        style.configure(
            "TButton",
            background=self.colors["surface_alt"],
            foreground=self.colors["text"],
            font=("Segoe UI", 10),
            padding=(10, 6),
            relief="raised",
            borderwidth=1,
        )
        style.map("TButton", background=[("active", self.colors["highlight"])])
        style.configure("TEntry", padding=(7, 5))
        style.configure("TCombobox", padding=(7, 5))
        style.configure("TSpinbox", padding=(7, 5))
        style.configure("TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#dde7f0",
            foreground=self.colors["text"],
            padding=(18, 9),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#edf3f8")],
            foreground=[("selected", self.colors["text"])],
        )
        style.configure("Dialog.TNotebook", background=self.colors["bg"], borderwidth=0)
        style.configure(
            "Dialog.TNotebook.Tab",
            background="#dde7f0",
            foreground=self.colors["text"],
            padding=(14, 7),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Dialog.TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#edf3f8")],
            padding=[("selected", (22, 11)), ("!selected", (14, 7))],
        )
        style.configure(
            "Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground=self.colors["text"],
            rowheight=30,
            font=("Segoe UI", 9),
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background=self.colors["surface_alt"],
            foreground=self.colors["text"],
            font=("Segoe UI", 9, "bold"),
            padding=(8, 6),
        )
        style.configure("Horizontal.TProgressbar", thickness=13)

    def _surface(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=16, style="Surface.TFrame", relief=tk.RAISED, borderwidth=1)
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
            text="入力、保存先、出力プロファイル、実行状況を1画面で確認し、詳細だけを別ウィンドウで調整します。",
            style="Subtitle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        workspace = ttk.Frame(main, style="App.TFrame")
        workspace.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        self.workspace = workspace
        self.workspace_layout = "wide"
        workspace.columnconfigure(0, weight=2, minsize=440)
        workspace.columnconfigure(1, weight=3, minsize=520)
        workspace.rowconfigure(0, weight=1)
        workspace.bind("<Configure>", self._on_workspace_configure, add="+")

        profile_shell = ttk.Frame(workspace, style="App.TFrame")
        self.profile_shell = profile_shell
        profile_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        run_shell = ttk.Frame(workspace, style="App.TFrame")
        self.run_shell = run_shell
        run_shell.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.profile_tab_body = self._scrollable_tab(profile_shell)
        self.run_tab_body = self._scrollable_tab(run_shell)

        self._build_profile_tab()
        self._build_run_tab()

    def _on_workspace_configure(self, event: tk.Event) -> None:
        self._apply_workspace_layout(int(getattr(event, "width", 0) or 0))

    def _apply_workspace_layout(self, width: int) -> None:
        if not hasattr(self, "workspace"):
            return
        layout = "stacked" if width and width < WORKSPACE_STACK_WIDTH else "wide"
        if layout == getattr(self, "workspace_layout", None):
            return

        self.workspace_layout = layout
        self.profile_shell.grid_forget()
        self.run_shell.grid_forget()
        if layout == "stacked":
            self.workspace.columnconfigure(0, weight=1, minsize=0)
            self.workspace.columnconfigure(1, weight=0, minsize=0)
            self.workspace.rowconfigure(0, weight=1)
            self.workspace.rowconfigure(1, weight=1)
            self.profile_shell.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 8))
            self.run_shell.grid(row=1, column=0, sticky="nsew", padx=0, pady=(8, 0))
        else:
            self.workspace.columnconfigure(0, weight=2, minsize=440)
            self.workspace.columnconfigure(1, weight=3, minsize=520)
            self.workspace.rowconfigure(0, weight=1)
            self.workspace.rowconfigure(1, weight=0)
            self.profile_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
            self.run_shell.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)

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
        ttk.Button(row, text="参照", command=lambda: self.browse_dir(variable)).grid(
            row=0, column=2, sticky=tk.E, padx=(8, 0)
        )

    def _menu_button(self, parent: ttk.Frame, text: str, items: List[tuple[str, object]]) -> ttk.Menubutton:
        button = ttk.Menubutton(parent, text=text)
        menu = tk.Menu(button, tearoff=False)
        for label, command in items:
            if label == "-":
                menu.add_separator()
            else:
                menu.add_command(label=label, command=command)
        button.configure(menu=menu)
        return button

    def _backend_display_name(self, backend: str) -> str:
        return BACKEND_DISPLAY_NAMES.get(backend, backend.upper())

    def _brand_badge(self, parent: tk.Misc, backend: str) -> tk.Label:
        text, bg, fg = BRAND_COLORS.get(backend, (backend.upper(), self.colors["surface_alt"], self.colors["text"]))
        return tk.Label(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            padx=9,
            pady=3,
            font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT,
        )

    def selected_resource_labels(self, profile: EncodeProfile, selected: List[str]) -> List[str]:
        resources = {resource.id: resource for resource in profile.hardware_resources}
        return [resources[resource_id].label if resource_id in resources else resource_id for resource_id in selected]

    def _widget_exists(self, widget: object) -> bool:
        if widget is None:
            return False
        exists = getattr(widget, "winfo_exists", None)
        if exists is None:
            return True
        try:
            return bool(exists())
        except tk.TclError:
            return False

    def _configure_widget_state(self, widget: object, enabled: bool) -> None:
        if self._widget_exists(widget):
            widget.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def _set_grid_pair_visible(self, widget: object, visible: bool) -> None:
        if not self._widget_exists(widget):
            return
        label = getattr(widget, "_label_widget", None)
        for item in (label, widget):
            if not self._widget_exists(item):
                continue
            if visible:
                grid = getattr(item, "grid", None)
                if grid is not None:
                    grid()
            else:
                grid_remove = getattr(item, "grid_remove", None)
                if grid_remove is not None:
                    grid_remove()

    def _build_run_tab(self) -> None:
        tab = self.run_tab_body
        summary = self._surface(tab)
        summary.pack(fill=tk.X)
        self.run_summary_frame = summary
        self.run_summary_layout = "wide"
        summary.bind("<Configure>", self._on_run_summary_configure, add="+")

        left = ttk.Frame(summary, style="Surface.TFrame")
        self.run_summary_left = left
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(left, textvariable=self.status_var, style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(left, textvariable=self.input_summary_var, style="Muted.TLabel").pack(anchor=tk.W, pady=(6, 0))
        ttk.Label(left, textvariable=self.output_summary_var, style="Muted.TLabel").pack(anchor=tk.W)

        controls = ttk.Frame(summary, style="Surface.TFrame")
        self.run_controls_frame = controls
        self.run_controls_layout = ""
        controls.pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Button(controls, text="開始", style="Accent.TButton", command=self.start_current_profile).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.pause_button = ttk.Button(controls, text="一時停止", command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="停止", style="Danger.TButton", command=self.stop_all).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(controls, text="再開", command=self.resume_saved).pack(side=tk.LEFT)

        control_children = controls.winfo_children()
        self.start_button = control_children[0]
        self.pause_button = control_children[1]
        self.stop_button = control_children[2]
        self.resume_button = control_children[3]
        self.run_control_buttons = [
            self.start_button,
            self.pause_button,
            self.stop_button,
            self.resume_button,
        ]
        for button in self.run_control_buttons:
            button.pack_forget()
        self._layout_run_controls(False)
        self._sync_runtime_controls()

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
        self._menu_button(
            list_head,
            "管理",
            [
                ("再スキャン", self.scan_files),
                ("ログフォルダを開く", self.open_log_dir),
            ],
        ).pack(side=tk.RIGHT)

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

    def _on_run_summary_configure(self, event: tk.Event) -> None:
        width = int(getattr(event, "width", 0) or 0)
        self._layout_run_summary(width < RUN_SUMMARY_STACK_WIDTH)
        self._layout_run_controls(width < RUN_CONTROLS_STACK_WIDTH)
        self._sync_runtime_controls()

    def _layout_run_summary(self, stacked: bool) -> None:
        if not hasattr(self, "run_summary_left") or not hasattr(self, "run_controls_frame"):
            return
        layout = "stacked" if stacked else "wide"
        if layout == getattr(self, "run_summary_layout", None):
            return
        self.run_summary_layout = layout
        self.run_summary_left.pack_forget()
        self.run_controls_frame.pack_forget()
        if stacked:
            self.run_summary_left.pack(fill=tk.X)
            self.run_controls_frame.pack(fill=tk.X, pady=(12, 0))
        else:
            self.run_summary_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.run_controls_frame.pack(side=tk.RIGHT, padx=(12, 0))

    def _layout_run_controls(self, stacked: bool) -> None:
        if not hasattr(self, "run_control_buttons"):
            return
        layout = "stacked" if stacked else "wide"
        if layout == getattr(self, "run_controls_layout", None):
            return
        self.run_controls_layout = layout
        for button in self.run_control_buttons:
            button.grid_forget()
        if stacked:
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for index in range(2):
                self.run_controls_frame.columnconfigure(index, weight=1, uniform="run-controls")
            for button, (row, column) in zip(self.run_control_buttons, positions):
                button.grid(row=row, column=column, sticky="ew", padx=4, pady=4)
        else:
            for index in range(4):
                self.run_controls_frame.columnconfigure(index, weight=0, uniform="")
            for column, button in enumerate(self.run_control_buttons):
                button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0), pady=0)

    def _sync_runtime_controls(self) -> None:
        if not hasattr(self, "start_button"):
            return
        busy = self.running or self.preparing
        self.start_button.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.resume_button.configure(state=tk.DISABLED if busy else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if self.running else tk.DISABLED)
        self.pause_button.configure(
            text="再開" if self.paused else "一時停止",
            state=tk.NORMAL if self.running else tk.DISABLED,
        )
        if self.running:
            self.pause_button.grid()
        else:
            self.pause_button.grid_remove()

    def _build_profile_tab(self) -> None:
        tab = self.profile_tab_body
        form = self._surface(tab)
        form.pack(fill=tk.X)

        toolbar = ttk.Frame(form, style="Surface.TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(toolbar, text="設定", style="Section.TLabel").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="保存", style="Accent.TButton", command=self.save_current_profile).pack(side=tk.RIGHT)

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
        self._path_row(general, "入力先", self.input_dir_var)
        self._path_row(general, "出力先", self.output_dir_var)
        self._path_row(general, "処理済みソース退避先", self.archive_dir_var)

        outputs = self._surface(tab)
        outputs.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        output_head = ttk.Frame(outputs, style="Surface.TFrame")
        output_head.pack(fill=tk.X)
        ttk.Label(output_head, text="出力プロファイル", style="Section.TLabel").pack(side=tk.LEFT)
        self._menu_button(
            output_head,
            "出力操作",
            [
                ("新規作成", self.open_new_output_dialog),
                ("詳細編集", self.edit_selected_output),
                ("有効化", lambda: self.set_selected_outputs_enabled(True)),
                ("無効化", lambda: self.set_selected_outputs_enabled(False)),
                ("複製", self.duplicate_selected_outputs),
                ("削除", self.remove_output),
            ],
        ).pack(side=tk.RIGHT)

        output_body = ttk.Frame(outputs, style="Surface.TFrame")
        output_body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        tree_frame = ttk.Frame(output_body, style="Surface.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        columns = ("enabled", "name", "backend", "encoder", "resolution", "folder", "container")
        self.outputs_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=7, selectmode="extended")
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
        self.outputs_tree.bind("<Double-1>", lambda _event: self.edit_selected_output())
        self.outputs_tree.bind("<Button-3>", self.on_outputs_tree_context)
        self.outputs_tree.tag_configure(BACKEND_CPU, background="#f1efff")
        self.outputs_tree.tag_configure(BACKEND_NVENC, background="#eef8dc")
        self.outputs_tree.tag_configure(BACKEND_QSV, background="#e6f3ff")
        self.outputs_tree.tag_configure(BACKEND_AMF, background="#ffecec")

        self.output_context_menu = tk.Menu(self.outputs_tree, tearoff=False)
        self.output_context_menu.add_command(label="詳細編集", command=self.edit_selected_output)
        self.output_context_menu.add_separator()
        self.output_context_menu.add_command(label="有効化", command=lambda: self.set_selected_outputs_enabled(True))
        self.output_context_menu.add_command(label="無効化", command=lambda: self.set_selected_outputs_enabled(False))
        self.output_context_menu.add_command(label="複製", command=self.duplicate_selected_outputs)
        self.output_context_menu.add_separator()
        self.output_context_menu.add_command(label="削除", command=self.remove_output)

        self.output_name_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.output_resolution_var = tk.StringVar(value="1080p")
        self.output_custom_height_var = tk.StringVar(value="")
        self.output_container_var = tk.StringVar(value="mp4")
        self.output_enabled_var = tk.BooleanVar(value=True)
        self.output_filename_template_var = tk.StringVar(value="{source}")
        self.output_input_dir_var = tk.StringVar(value="")
        self.output_output_dir_var = tk.StringVar(value="")
        self.output_segment_minutes_var = tk.StringVar(value="")
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

        editor_actions = ttk.Frame(output_body, style="Surface.TFrame")
        editor_actions.pack(fill=tk.X, pady=(10, 0))
        self.output_detail_var = tk.StringVar(value="出力プロファイルを選択するか、新規作成してください。")
        ttk.Label(editor_actions, textvariable=self.output_detail_var, style="Muted.TLabel").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(editor_actions, text="新規", command=self.open_new_output_dialog).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(editor_actions, text="詳細編集", style="Accent.TButton", command=self.edit_selected_output).pack(
            side=tk.RIGHT
        )

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
        setattr(entry, "_label_widget", label_widget)
        return label_widget, entry

    def browse_dir(self, variable: tk.StringVar) -> None:
        value = filedialog.askdirectory(initialdir=variable.get() or str(self.paths.base_dir))
        if value:
            variable.set(value)

    def _profile_labels(self) -> List[str]:
        return [self.profile_display_label(profile) for profile in self.profiles]

    def profile_short_id(self, profile: EncodeProfile) -> str:
        return profile.id.rsplit("_", 1)[-1][:8]

    def profile_display_label(self, profile: EncodeProfile) -> str:
        input_label = Path(profile.input_dir).name or "入力未設定"
        output_label = Path(profile.output_dir).name or "出力未設定"
        return f"{input_label} -> {output_label} ({self.profile_short_id(profile)})"

    def refresh_profile_choices(self) -> None:
        if not hasattr(self, "profile_combo"):
            if self.profiles and self.active_profile_id not in {profile.id for profile in self.profiles}:
                self.active_profile_id = self.profiles[0].id
            return
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
            ttk.Label(self.resource_frame, text="slots", style="Surface.TLabel").grid(
                row=row, column=1, sticky=tk.W, padx=(16, 4)
            )
            ttk.Spinbox(self.resource_frame, from_=1, to=16, textvariable=slot_var, width=5).grid(
                row=row, column=2, sticky=tk.W, pady=2
            )
            if resource.detected_encoder_engines:
                detail = f"detected NVENC: {resource.detected_encoder_engines}"
            elif resource.detection_error:
                detail = "NVENCエンジン数未検出、slotsは手動調整"
            else:
                detail = ""
            if detail:
                ttk.Label(self.resource_frame, text=detail, style="Muted.TLabel").grid(
                    row=row, column=3, sticky=tk.W, padx=(10, 0)
                )
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
        if hasattr(self, "profiles") and self.profiles:
            return [resource.id for resource in self.current_profile().hardware_resources]
        return [resource_id for resource_id, var in getattr(self, "resource_enabled_vars", {}).items() if var.get()]

    def _encoders_for_backend(
        self,
        backend: str,
        resource_ids: Optional[List[str]] = None,
        verify_nvenc: bool = False,
    ) -> List[str]:
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
                if backend == BACKEND_NVENC and resource_ids:
                    encoders = [
                        encoder
                        for encoder in encoders
                        if self.nvenc_encoder_supported_by_resources(encoder, resource_ids, verify=verify_nvenc)
                    ]
                return sorted(encoders)
        if backend == BACKEND_NVENC:
            if resource_ids and verify_nvenc:
                return [
                    encoder
                    for encoder in GPU_CODECS
                    if self.nvenc_encoder_supported_by_resources(encoder, resource_ids, verify=True)
                ]
            return GPU_CODECS
        if backend == BACKEND_QSV:
            return ["h264_qsv", "hevc_qsv", "av1_qsv"]
        if backend == BACKEND_AMF:
            return ["h264_amf", "hevc_amf", "av1_amf"]
        return CPU_CODECS

    def selected_output_resource_ids(self) -> List[str]:
        selected = [resource_id for resource_id, var in self.output_resource_vars.items() if var.get()]
        return selected

    def smoke_test_encoder_cached(
        self, encoder: str, resource_id: str = "", split_encode_mode: str = ""
    ) -> tuple[bool, str]:
        if not hasattr(self, "encoder_smoke_cache"):
            self.encoder_smoke_cache = {}
        key = (encoder, resource_id, split_encode_mode)
        if key not in self.encoder_smoke_cache:
            self.encoder_smoke_cache[key] = smoke_test_encoder(
                self.paths.ffmpeg_path,
                encoder,
                resource_id=resource_id,
                split_encode_mode=split_encode_mode,
                timeout=30,
            )
        return self.encoder_smoke_cache[key]

    def nvenc_encoder_supported_by_resources(self, encoder: str, resource_ids: List[str], verify: bool = False) -> bool:
        if not encoder.endswith("_nvenc"):
            return True
        nvenc_resources = [
            resource_id for resource_id in resource_ids if resource_backend(resource_id) == BACKEND_NVENC
        ]
        if not nvenc_resources:
            return False
        if not verify:
            return True
        return all(self.smoke_test_encoder_cached(encoder, resource_id)[0] for resource_id in nvenc_resources)

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
        resources = list(profile.hardware_resources)
        if not resources:
            ttk.Label(self.output_resource_frame, text="利用できるリソースがありません", style="Inset.TLabel").grid(
                row=0, column=0, sticky=tk.W
            )
            return

        selected = [
            resource_id for resource_id in selected if any(resource.id == resource_id for resource in resources)
        ]
        selected_backend = resource_backend(selected[0]) if selected else backend
        compatible_ids = [resource.id for resource in resources if resource_backend(resource.id) == selected_backend]
        if not compatible_ids:
            selected_backend = resource_backend(resources[0].id)
            compatible_ids = [
                resource.id for resource in resources if resource_backend(resource.id) == selected_backend
            ]
        selected = select_compatible_resource_ids(compatible_ids, selected)
        if selected:
            selected_backend = resource_backend(selected[0])
            self.output_backend_var.set(selected_backend)

        for resource in resources:
            var = tk.BooleanVar(value=resource.id in selected)
            self.output_resource_vars[resource.id] = var

        labels = self.selected_resource_labels(profile, selected)
        if selected:
            self._brand_badge(self.output_resource_frame, selected_backend).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(
            self.output_resource_frame,
            text=f"{self._backend_display_name(selected_backend)} / {', '.join(labels)}",
            style="Inset.TLabel",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def open_output_resource_dialog(self) -> None:
        existing = getattr(self, "output_resource_window", None)
        if self._widget_exists(existing):
            existing.lift()
            return

        profile = self.current_profile()
        resources = list(profile.hardware_resources)
        if not resources:
            messagebox.showerror("入力エラー", "利用できるリソースがありません。")
            return

        dialog = tk.Toplevel(self.root)
        self.output_resource_window = dialog
        dialog.title("エンコード先")
        dialog.transient(self.root)
        dialog.configure(bg=self.colors["bg"])
        dialog.resizable(False, False)

        def on_close() -> None:
            self.output_resource_window = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)

        body = ttk.Frame(dialog, padding=16, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="エンコード先", style="Section.TLabel").pack(anchor=tk.W, pady=(0, 10))

        selected = set(self.selected_output_resource_ids())
        if not selected:
            selected = set(select_compatible_resource_ids([item.id for item in resources], []))
        resource_vars = {resource.id: tk.BooleanVar(value=resource.id in selected) for resource in resources}

        def on_change(changed_id: str) -> None:
            if not resource_vars[changed_id].get():
                return
            backend = resource_backend(changed_id)
            for resource_id, var in resource_vars.items():
                if resource_id != changed_id and resource_backend(resource_id) != backend:
                    var.set(False)

        grouped: Dict[str, ttk.Frame] = {}
        for resource in resources:
            backend = resource_backend(resource.id)
            if backend not in grouped:
                group = ttk.Frame(body, padding=10, style="Surface.TFrame", relief=tk.RAISED, borderwidth=1)
                group.pack(fill=tk.X, pady=(0, 10))
                head = ttk.Frame(group, style="Surface.TFrame")
                head.pack(fill=tk.X, pady=(0, 6))
                self._brand_badge(head, backend).pack(side=tk.LEFT, padx=(0, 8))
                ttk.Label(head, text=self._backend_display_name(backend), style="Surface.TLabel").pack(side=tk.LEFT)
                grouped[backend] = group
            row = ttk.Frame(grouped[backend], style="Surface.TFrame")
            row.pack(fill=tk.X, pady=2)
            ttk.Checkbutton(
                row,
                text=resource.label,
                variable=resource_vars[resource.id],
                command=lambda resource_id=resource.id: on_change(resource_id),
            ).pack(side=tk.LEFT)
            detail = f"{resource.concurrency_slots} slot(s)"
            if resource.detected_encoder_engines:
                detail = f"NVENC {resource.detected_encoder_engines} engine(s)"
            elif resource.detection_error:
                detail = "NVENCエンジン数未検出、slotsは手動調整"
            ttk.Label(row, text=detail, style="Muted.TLabel").pack(side=tk.RIGHT)

        buttons = ttk.Frame(body, style="App.TFrame")
        buttons.pack(fill=tk.X, pady=(4, 0))

        def add_manual_and_close() -> None:
            on_close()
            self.add_manual_resource()

        def apply_selection() -> None:
            selected_ids = [resource_id for resource_id, var in resource_vars.items() if var.get()]
            if not selected_ids:
                messagebox.showerror("入力エラー", "リソースを1つ以上選択してください。")
                return
            backend = resource_backend(selected_ids[0])
            if any(resource_backend(resource_id) != backend for resource_id in selected_ids):
                messagebox.showerror("入力エラー", "1つの出力内で異なるBackendのリソースは混在できません。")
                return
            self.output_resource_vars = {
                resource.id: tk.BooleanVar(value=resource.id in selected_ids) for resource in resources
            }
            self.output_backend_var.set(backend)
            self.update_output_encoder_controls()
            on_close()

        ttk.Button(buttons, text="手動追加", command=add_manual_and_close).pack(side=tk.LEFT)
        ttk.Button(buttons, text="適用", style="Accent.TButton", command=apply_selection).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="閉じる", command=on_close).pack(side=tk.RIGHT, padx=(0, 8))

    def _dialog_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        row: int,
        pair: int = 0,
    ) -> ttk.Entry:
        label_column = pair * 2
        label_widget = ttk.Label(parent, text=label, style="Surface.TLabel")
        label_widget.grid(
            row=row,
            column=label_column,
            sticky=tk.W,
            pady=4,
            padx=(0 if pair == 0 else 16, 8),
        )
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=label_column + 1, sticky="ew", pady=4)
        setattr(entry, "_label_widget", label_widget)
        return entry

    def _dialog_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        values: List[str],
        row: int,
        pair: int = 0,
        width: int = 18,
    ) -> ttk.Combobox:
        label_column = pair * 2
        label_widget = ttk.Label(parent, text=label, style="Surface.TLabel")
        label_widget.grid(
            row=row,
            column=label_column,
            sticky=tk.W,
            pady=4,
            padx=(0 if pair == 0 else 16, 8),
        )
        combo = ttk.Combobox(parent, textvariable=variable, values=values, width=width, state="readonly")
        combo.grid(row=row, column=label_column + 1, sticky="ew", pady=4)
        setattr(combo, "_label_widget", label_widget)
        return combo

    def _dialog_search_combo(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.Variable,
        values: List[str],
        row: int,
        pair: int = 0,
        width: int = 18,
    ) -> SearchableCombobox:
        label_column = pair * 2
        label_widget = ttk.Label(parent, text=label, style="Surface.TLabel")
        label_widget.grid(
            row=row,
            column=label_column,
            sticky=tk.W,
            pady=4,
            padx=(0 if pair == 0 else 16, 8),
        )
        combo = SearchableCombobox(parent, textvariable=variable, values=values, width=width, state="normal")
        combo.grid(row=row, column=label_column + 1, sticky="ew", pady=4)
        setattr(combo, "_label_widget", label_widget)
        return combo

    def _dialog_surface(self, parent: ttk.Frame, title: str) -> ttk.Frame:
        frame = ttk.Frame(parent, padding=12, style="Surface.TFrame", relief=tk.RAISED, borderwidth=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        ttk.Label(frame, text=title, style="Section.TLabel").grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8)
        )
        return frame

    def open_output_advanced_dialog(self) -> None:
        existing = getattr(self, "output_advanced_window", None)
        if self._widget_exists(existing):
            existing.lift()
            return

        dialog = tk.Toplevel(self.root)
        self.output_advanced_window = dialog
        dialog.title("出力の詳細設定")
        dialog.transient(self.root)
        dialog.configure(bg=self.colors["bg"])
        dialog.geometry("720x560")

        def on_close() -> None:
            self.output_advanced_window = None
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)

        body = ttk.Frame(dialog, padding=14, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        tabs = ttk.Notebook(body, style="Dialog.TNotebook")
        tabs.pack(fill=tk.BOTH, expand=True)

        video_tab = ttk.Frame(tabs, padding=10, style="App.TFrame")
        audio_tab = ttk.Frame(tabs, padding=10, style="App.TFrame")
        ffmpeg_tab = ttk.Frame(tabs, padding=10, style="App.TFrame")
        tabs.add(video_tab, text="映像")
        tabs.add(audio_tab, text="音声")
        tabs.add(ffmpeg_tab, text="FFmpeg")

        self.output_engine_panel_container = ttk.Frame(video_tab, style="App.TFrame")
        self.output_engine_panel_container.pack(fill=tk.X)

        self.output_nvenc_detail_frame = self._dialog_surface(self.output_engine_panel_container, "NVENC")
        self.output_preset_combo = self._dialog_combo(
            self.output_nvenc_detail_frame, "NVENC Preset", self.output_preset_var, GPU_PRESETS, 1, 0
        )
        self.output_tune_combo = self._dialog_combo(
            self.output_nvenc_detail_frame, "NVENC Tune", self.output_tune_var, GPU_TUNES, 1, 1
        )
        self.output_split_combo = self._dialog_combo(
            self.output_nvenc_detail_frame, "SFE", self.output_split_encode_mode_var, ["auto"], 2, 0, width=18
        )

        self.output_cpu_detail_frame = self._dialog_surface(self.output_engine_panel_container, "CPU")
        self.output_cpu_preset_combo = self._dialog_combo(
            self.output_cpu_detail_frame, "CPU Preset", self.output_cpu_preset_var, CPU_PRESETS, 1, 0
        )
        self.output_cpu_tune_combo = self._dialog_combo(
            self.output_cpu_detail_frame,
            "CPU Tune",
            self.output_cpu_tune_var,
            CPU_TUNES_BY_CODEC.get(self.output_cpu_codec_var.get(), CPU_TUNES_BY_CODEC["libx264"]),
            1,
            1,
        )

        self.output_hardware_detail_frame = self._dialog_surface(self.output_engine_panel_container, "Hardware")
        ttk.Label(
            self.output_hardware_detail_frame,
            text="QSV/AMF固有の調整はFFmpeg引数タブで指定できます。",
            style="Surface.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky=tk.W)

        rate = self._dialog_surface(video_tab, "レート制御")
        rate.pack(fill=tk.X, pady=(10, 0))
        self.output_maxrate_entry = self._dialog_entry(rate, "Maxrate", self.output_maxrate_var, 1, 0)
        self.output_bufsize_entry = self._dialog_entry(rate, "Bufsize", self.output_bufsize_var, 1, 1)

        format_box = self._dialog_surface(video_tab, "フォーマット")
        format_box.pack(fill=tk.X, pady=(10, 0))
        self._dialog_entry(format_box, "分割間隔(分・空欄=基本)", self.output_segment_minutes_var, 1, 0)
        self._dialog_entry(format_box, "Pix fmt", self.output_pix_fmt_var, 1, 1)
        self._dialog_entry(format_box, "Scale flags", self.output_scale_flags_var, 2, 0)

        audio = self._dialog_surface(audio_tab, "音声")
        audio.pack(fill=tk.X)
        self.output_audio_codec_combo = self._dialog_search_combo(
            audio,
            "Audio codec",
            self.output_audio_codec_var,
            AUDIO_CODEC_CHOICES,
            1,
            0,
        )
        self.output_audio_codec_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_output_audio_container_choices()
        )
        self._dialog_entry(audio, "Audio bitrate", self.output_audio_bitrate_var, 1, 1)
        self.output_audio_container_combo = self._dialog_search_combo(
            audio,
            "Audio container",
            self.output_audio_container_var,
            audio_containers_for_codec(self.output_audio_codec_var.get()),
            2,
            0,
        )
        self.output_audio_container_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self.update_output_audio_container_choices()
        )

        ffmpeg = self._dialog_surface(ffmpeg_tab, "追加引数")
        ffmpeg.pack(fill=tk.X)
        self._dialog_entry(ffmpeg, "FFmpeg input args", self.output_extra_input_args_var, 1, 0)
        self._dialog_entry(ffmpeg, "FFmpeg video args", self.output_extra_video_args_var, 1, 1)
        self._dialog_entry(ffmpeg, "FFmpeg audio args", self.output_extra_audio_args_var, 2, 0)
        self._dialog_entry(ffmpeg, "FFmpeg output args", self.output_extra_output_args_var, 2, 1)
        self._dialog_entry(ffmpeg, "FFmpeg concat args", self.output_extra_concat_args_var, 3, 0)
        self._dialog_entry(ffmpeg, "FFmpeg mux args", self.output_extra_mux_args_var, 3, 1)

        bottom = ttk.Frame(body, style="App.TFrame")
        bottom.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(bottom, text="閉じる", style="Accent.TButton", command=on_close).pack(side=tk.RIGHT)

        self.update_output_cpu_tune_choices()
        self.update_output_audio_container_choices()
        self.update_output_encoder_controls()

    def _sync_output_advanced_panels(self, backend: str) -> None:
        panels = [
            getattr(self, "output_nvenc_detail_frame", None),
            getattr(self, "output_cpu_detail_frame", None),
            getattr(self, "output_hardware_detail_frame", None),
        ]
        for panel in panels:
            if self._widget_exists(panel):
                panel.pack_forget()
        if backend == BACKEND_CPU:
            panel = getattr(self, "output_cpu_detail_frame", None)
        elif backend == BACKEND_NVENC:
            panel = getattr(self, "output_nvenc_detail_frame", None)
        else:
            panel = getattr(self, "output_hardware_detail_frame", None)
        if self._widget_exists(panel):
            panel.pack(fill=tk.X)

    def add_manual_resource(self) -> None:
        backend_text = simpledialog.askstring(
            "手動リソース追加",
            "Backend を入力してください: nvidia / intel / amd",
            parent=self.root,
        )
        if backend_text is None:
            return
        backend_key = backend_text.strip().lower()
        backend = {
            "nvidia": BACKEND_NVENC,
            "nvenc": BACKEND_NVENC,
            "intel": BACKEND_QSV,
            "qsv": BACKEND_QSV,
            "amd": BACKEND_AMF,
            "amf": BACKEND_AMF,
        }.get(backend_key)
        if backend is None:
            messagebox.showerror("入力エラー", "Backend は nvidia / intel / amd のいずれかを入力してください。")
            return

        index = simpledialog.askinteger("手動リソース追加", "GPU index", parent=self.root, minvalue=0, initialvalue=0)
        if index is None:
            return
        default_label = {
            BACKEND_NVENC: f"NVIDIA GPU {index} (manual)",
            BACKEND_QSV: f"Intel GPU {index} (manual)",
            BACKEND_AMF: f"AMD GPU {index} (manual)",
        }[backend]
        label = simpledialog.askstring("手動リソース追加", "表示名", parent=self.root, initialvalue=default_label)
        if label is None:
            return
        slots = simpledialog.askinteger(
            "手動リソース追加", "スロット数", parent=self.root, minvalue=1, maxvalue=16, initialvalue=1
        )
        if slots is None:
            return

        resource_id = resource_id_for_backend(backend, index)
        vendor = {"nvidia": "nvidia", "intel": "intel", "amd": "amd"}[resource_id.split(":", 1)[0]]
        resource = HardwareResource(
            id=resource_id,
            label=label.strip() or default_label,
            kind="gpu",
            backend=backend,
            vendor=vendor,
            index=index,
            concurrency_slots=slots,
            detection_error="Manual fallback resource; availability is checked before encoding.",
        )

        profile = self.current_profile()
        profile.hardware_resources = [item for item in profile.hardware_resources if item.id != resource.id]
        profile.hardware_resources.append(resource)
        if resource.id not in profile.resource_ids:
            profile.resource_ids.append(resource.id)
        self.output_backend_var.set(backend)
        self.render_output_resource_controls(profile, backend, [resource.id])
        self.update_output_encoder_controls()

    def update_output_encoder_controls(self) -> None:
        if not self._widget_exists(getattr(self, "output_encoder_combo", None)):
            return
        backend = self.output_backend_var.get() or BACKEND_CPU
        selected = self.selected_output_resource_ids()
        self.render_output_resource_controls(self.current_profile(), backend, selected)
        selected = self.selected_output_resource_ids()
        backend = self.output_backend_var.get() or backend
        encoders = self._encoders_for_backend(backend, selected, verify_nvenc=False)
        self.output_encoder_combo.configure(values=encoders)
        if self.output_encoder_var.get() not in encoders:
            self.output_encoder_var.set(encoders[0] if encoders else "")

        encoder = self.output_encoder_var.get()
        self.output_codec_var.set(encoder if backend == BACKEND_NVENC else self.output_codec_var.get())
        self.output_cpu_codec_var.set(encoder if backend == BACKEND_CPU else self.output_cpu_codec_var.get())
        self.update_output_cpu_tune_choices()
        rate_modes = rate_modes_for_backend(backend)
        if hasattr(self, "output_rate_combo"):
            self.output_rate_combo.configure(values=rate_modes)
        if self.output_rate_mode_var.get().upper() not in rate_modes:
            self.output_rate_mode_var.set(rate_modes[0])
        self.update_output_rate_controls()
        self._sync_output_advanced_panels(backend)

        nvenc_state = tk.NORMAL if backend == BACKEND_NVENC else tk.DISABLED
        cpu_state = tk.NORMAL if backend == BACKEND_CPU else tk.DISABLED
        for widget in (getattr(self, "output_preset_combo", None), getattr(self, "output_tune_combo", None)):
            if self._widget_exists(widget):
                widget.configure(state=nvenc_state)
        for widget in (
            getattr(self, "output_cpu_preset_combo", None),
            getattr(self, "output_cpu_tune_combo", None),
        ):
            if self._widget_exists(widget):
                widget.configure(state=cpu_state)

        split_modes = self.split_encode_modes_for_encoder(encoder)
        split_combo = getattr(self, "output_split_combo", None)
        if backend == BACKEND_NVENC and encoder in {"hevc_nvenc", "av1_nvenc"} and split_modes:
            if self._widget_exists(split_combo):
                split_combo.configure(values=split_modes, state=tk.NORMAL)
                self._set_grid_pair_visible(split_combo, True)
        else:
            split_modes = ["auto"]
            self.output_split_encode_mode_var.set("auto")
            if self._widget_exists(split_combo):
                split_combo.configure(values=["auto"], state=tk.NORMAL)
                self._set_grid_pair_visible(split_combo, False)
        if self.output_split_encode_mode_var.get() not in split_modes:
            self.output_split_encode_mode_var.set(split_modes[0])

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

        if hasattr(self, "profile_combo"):
            combo_index = self.profile_combo.current()
            if 0 <= combo_index < len(self.profiles):
                self.active_profile_id = self.profiles[combo_index].id
                return self.profiles[combo_index]

        self.active_profile_id = self.profiles[0].id
        return self.profiles[0]

    def on_profile_selected(self, _event: object = None) -> None:
        if not hasattr(self, "profile_combo"):
            return
        combo_index = self.profile_combo.current()
        if 0 <= combo_index < len(self.profiles):
            self.active_profile_id = self.profiles[combo_index].id
        if self.running:
            self.log("実行中のため、表示プロファイルだけ切り替えます。")
        self.load_profile_into_form(self.current_profile())
        self.scan_files()

    def load_profile_into_form(self, profile: EncodeProfile) -> None:
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
        self.output_input_dir_var.set("")
        self.output_output_dir_var.set("")
        self.output_segment_minutes_var.set("")
        default_backend = default_output_backend_for_resource_ids(profile.resource_ids)
        self.output_backend_var.set(default_backend)
        self.output_encoder_var.set(
            DEFAULT_ENCODER_BY_BACKEND.get(default_backend, DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU])
        )
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
        self.output_audio_container_var.set("mka")
        self.output_extra_input_args_var.set("")
        self.output_extra_video_args_var.set("")
        self.output_extra_audio_args_var.set("")
        self.output_extra_output_args_var.set("")
        self.output_extra_concat_args_var.set("")
        self.output_extra_mux_args_var.set("")
        self.output_resource_vars = {}
        self.update_output_cpu_tune_choices()
        self.update_output_audio_container_choices()
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
        combo = getattr(self, "output_cpu_tune_combo", None)
        if not self._widget_exists(combo):
            return
        values = CPU_TUNES_BY_CODEC.get(self.output_cpu_codec_var.get(), CPU_TUNES_BY_CODEC["libx264"])
        combo.configure(values=values)
        if self.output_cpu_tune_var.get() not in values:
            self.output_cpu_tune_var.set("none")

    def update_output_audio_container_choices(self) -> None:
        codec = normalize_audio_codec(self.output_audio_codec_var.get())
        self.output_audio_codec_var.set(codec)
        values = audio_containers_for_codec(codec)
        combo = getattr(self, "output_audio_container_combo", None)
        if self._widget_exists(combo):
            if hasattr(combo, "set_values"):
                combo.set_values(values)
            else:
                combo.configure(values=values)
        self.output_audio_container_var.set(
            normalize_audio_container_for_codec(codec, self.output_audio_container_var.get())
        )

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
        for widget, visible in controls.items():
            self._set_grid_pair_visible(widget, visible)

    def update_output_rate_controls(self) -> None:
        if not self._widget_exists(getattr(self, "output_cq_entry", None)):
            return
        mode = self.output_rate_mode_var.get().upper()
        backend = self.output_backend_var.get() if hasattr(self, "output_backend_var") else BACKEND_CPU
        controls = [
            (getattr(self, "output_cq_entry", None), backend_allows_cq(backend) and mode in {"CQ", "VBR"}),
            (getattr(self, "output_bitrate_entry", None), mode in {"VBR", "ABR", "CBR"}),
            (getattr(self, "output_maxrate_entry", None), mode == "VBR"),
            (getattr(self, "output_bufsize_entry", None), mode in {"VBR", "CBR"}),
        ]
        for widget, visible in controls:
            self._set_grid_pair_visible(widget, visible)

    def update_resolution_controls(self) -> None:
        if not self._widget_exists(getattr(self, "custom_height_entry", None)):
            return
        if self.output_resolution_var.get() == "Custom":
            self.custom_height_label.grid()
            self.custom_height_entry.grid()
            self.custom_height_entry.configure(state=tk.NORMAL)
        else:
            self.output_custom_height_var.set("")
            self.custom_height_label.grid_remove()
            self.custom_height_entry.grid_remove()

    def collect_profile_from_form(self) -> Optional[EncodeProfile]:
        current = self.current_profile()
        mode = self.rate_mode_var.get().upper()
        if mode in {"CQ", "VBR"}:
            try:
                cq_value = int(self.cq_var.get())
            except ValueError:
                messagebox.showerror("入力エラー", "CQ/CRF は数値で入力してください。")
                return None
        else:
            cq_value = current.cq_value

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
        profile_resource_ids = [resource.id for resource in hardware_resources]
        first_gpu = next(
            (resource_id for resource_id in profile_resource_ids if resource_backend(resource_id) == BACKEND_NVENC), ""
        )
        use_gpu = profile_uses_nvenc_resource(profile_resource_ids)
        gpu_index = resource_index(first_gpu) if first_gpu else 0
        gpu_name = ""
        for resource in hardware_resources:
            if resource.id == first_gpu:
                gpu_name = resource.label
                break
        profile = EncodeProfile(
            id=current.id,
            input_dir=input_dir,
            output_dir=output_dir,
            archive_dir=archive_dir,
            max_parallel_jobs=current.max_parallel_jobs,
            segment_minutes=current.segment_minutes,
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
                messagebox.showerror(
                    "入力エラー",
                    f"{variant.name}: CQ は CPU/NVENC のみで使用できます。QSV/AMF では VBR/ABR/CBR を選択してください。",
                )
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

        for index, existing in enumerate(self.profiles):
            if existing.id == profile.id:
                self.profiles[index] = profile
                break
        else:
            self.profiles.append(profile)

        save_profiles(self.paths, self.profiles)
        self.active_profile_id = profile.id
        self.refresh_profile_choices()
        self.log(f"プロファイルを保存: {self.profile_display_label(profile)}")
        self.scan_files()

    def new_profile(self) -> None:
        profile = default_profile(self.paths, self.gpus)
        profile.id = new_id("profile")
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
        ok = messagebox.askyesno("削除", f"プロファイル '{self.profile_display_label(profile)}' を削除しますか？")
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
                tags=(backend,),
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

    def selected_output_ids(self) -> List[str]:
        selected = [str(item) for item in self.outputs_tree.selection()]
        if selected:
            return selected
        return [self.selected_output_id] if self.selected_output_id else []

    def on_outputs_tree_context(self, event: tk.Event) -> str:
        row_id = self.outputs_tree.identify_row(event.y)
        if row_id:
            selected = set(self.outputs_tree.selection())
            if row_id not in selected:
                self.outputs_tree.selection_set(row_id)
            self.selected_output_id = row_id
            self.on_output_select()
        try:
            self.output_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.output_context_menu.grab_release()
        return "break"

    def set_selected_outputs_enabled(self, enabled: bool) -> None:
        selected_ids = self.selected_output_ids()
        if not selected_ids:
            return
        ids = set(selected_ids)
        for variant in self.editing_outputs:
            if variant.id in ids:
                variant.enabled = enabled
        self.refresh_outputs_tree()
        self.outputs_tree.selection_set(*selected_ids)
        if self.selected_output_id in ids:
            self.on_output_select()

    def unique_output_name(self, base: str) -> str:
        existing = {variant.name for variant in self.editing_outputs}
        if base not in existing:
            return base
        index = 2
        while True:
            candidate = f"{base} {index}"
            if candidate not in existing:
                return candidate
            index += 1

    def unique_output_folder(self, base: str) -> str:
        existing = {variant.folder_name.lower() for variant in self.editing_outputs}
        root = safe_folder_name(base)
        if root.lower() not in existing:
            return root
        index = 2
        while True:
            candidate = safe_folder_name(f"{root}-{index}")
            if candidate.lower() not in existing:
                return candidate
            index += 1

    def duplicate_selected_outputs(self) -> None:
        ids = self.selected_output_ids()
        if not ids:
            return
        by_id = {variant.id: variant for variant in self.editing_outputs}
        added: List[str] = []
        for variant_id in ids:
            source = by_id.get(variant_id)
            if source is None:
                continue
            clone = OutputVariant.from_dict(asdict(source))
            clone.id = new_id("variant")
            clone.name = self.unique_output_name(f"{source.name} copy")
            clone.folder_name = self.unique_output_folder(f"{source.folder_name}-copy")
            self.editing_outputs.append(clone)
            added.append(clone.id)
        if not added:
            return
        self.refresh_outputs_tree()
        self.outputs_tree.selection_set(*added)
        self.selected_output_id = added[0]
        self.on_output_select()

    def on_output_select(self, _event: object = None) -> None:
        selection = self.outputs_tree.selection()
        if not selection:
            return
        if len(selection) > 1:
            self.selected_output_id = selection[0]
            if hasattr(self, "output_detail_var"):
                self.output_detail_var.set(f"{len(selection)}件選択中: 右クリックまたは出力操作から一括操作できます。")
            return
        self.selected_output_id = selection[0]
        profile = self.current_profile()
        for variant in self.editing_outputs:
            if variant.id == self.selected_output_id:
                if hasattr(self, "output_detail_var"):
                    state = "有効" if variant.enabled else "無効"
                    self.output_detail_var.set(
                        f"{variant.name} / {state} / {variant.ffmpeg_encoder or variant.codec or variant.cpu_codec}"
                    )
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
                self.output_input_dir_var.set(variant.input_dir)
                self.output_output_dir_var.set(variant.output_dir)
                self.output_segment_minutes_var.set(
                    "" if variant.segment_minutes is None else str(variant.segment_minutes)
                )
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
                self.output_audio_codec_var.set(normalize_audio_codec(variant.audio_codec or "copy"))
                self.output_audio_bitrate_var.set(variant.audio_bitrate)
                self.output_audio_container_var.set(
                    normalize_audio_container_for_codec(self.output_audio_codec_var.get(), variant.audio_container)
                )
                self.output_extra_input_args_var.set(variant.extra_input_args)
                self.output_extra_video_args_var.set(variant.extra_video_args)
                self.output_extra_audio_args_var.set(variant.extra_audio_args)
                self.output_extra_output_args_var.set(variant.extra_output_args)
                self.output_extra_concat_args_var.set(variant.extra_concat_args)
                self.output_extra_mux_args_var.set(variant.extra_mux_args)
                self.update_output_cpu_tune_choices()
                self.update_output_audio_container_choices()
                self.update_output_rate_controls()
                self.update_resolution_controls()
                self.render_output_resource_controls(profile, self.output_backend_var.get(), variant.resource_ids)
                self.update_output_encoder_controls()
                return

    def clear_output_selection(self) -> None:
        selection_remove = getattr(self.outputs_tree, "selection_remove", None)
        if selection_remove is not None:
            selection_remove(self.outputs_tree.selection())
        else:
            self.outputs_tree.selection_set()
        self.selected_output_id = None
        current_profile = getattr(self, "current_profile", None)
        if current_profile is not None and hasattr(self, "profiles") and hasattr(self, "active_profile_id"):
            self.set_output_edit_defaults(current_profile())
        if hasattr(self, "output_detail_var"):
            self.output_detail_var.set("出力プロファイルを選択するか、新規作成してください。")

    def open_new_output_dialog(self) -> None:
        self.clear_output_selection()
        self.open_output_editor_dialog()

    def edit_selected_output(self) -> None:
        selection = self.outputs_tree.selection()
        if len(selection) != 1:
            messagebox.showwarning("編集対象", "詳細編集は出力プロファイルを1つだけ選択してください。")
            return
        self.selected_output_id = selection[0]
        self.on_output_select()
        self.open_output_editor_dialog()

    def open_output_editor_dialog(self) -> None:
        profile = self.current_profile()
        variant = None
        if self.selected_output_id:
            for item in self.editing_outputs:
                if item.id == self.selected_output_id:
                    variant = item
                    break
        session = OutputEditorSession(self, profile, variant)
        session.show()

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

        raw_segment_minutes = self.output_segment_minutes_var.get().strip()
        segment_minutes: Optional[int] = None
        if raw_segment_minutes:
            try:
                segment_minutes = int(raw_segment_minutes)
            except ValueError:
                messagebox.showerror("入力エラー", "分割間隔は数値で入力してください。")
                return
            if segment_minutes < 1:
                messagebox.showerror("入力エラー", "分割間隔は1分以上にしてください。")
                return

        backend = self.output_backend_var.get() or BACKEND_CPU
        rate_mode = self.output_rate_mode_var.get().upper()
        if not backend_accepts_rate_mode(backend, rate_mode):
            messagebox.showerror(
                "入力エラー", "CQ は CPU/NVENC のみで使用できます。QSV/AMF では VBR/ABR/CBR を選択してください。"
            )
            return
        cq_value = (
            self._cq_value_for_rate_mode(rate_mode, self.output_cq_var.get(), self._output_cq_fallback())
            if backend_allows_cq(backend)
            else self._output_cq_fallback()
        )
        if cq_value is None:
            messagebox.showerror("入力エラー", "CQ/CRF は数値で入力してください。")
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
        audio_codec = normalize_audio_codec(self.output_audio_codec_var.get())
        audio_container = normalize_audio_container_for_codec(audio_codec, self.output_audio_container_var.get())
        variant = OutputVariant(
            id=self.selected_output_id or new_id("variant"),
            name=name,
            folder_name=folder_name,
            height=height,
            container=container,
            enabled=self.output_enabled_var.get() if hasattr(self, "output_enabled_var") else True,
            filename_template=self.output_filename_template_var.get().strip() or "{source}",
            input_dir=self.output_input_dir_var.get().strip(),
            output_dir=self.output_output_dir_var.get().strip(),
            segment_minutes=segment_minutes,
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
            audio_codec=audio_codec,
            audio_bitrate=self.output_audio_bitrate_var.get().strip(),
            audio_container=audio_container,
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
        existing_window = getattr(self, "output_editor_window", None)
        if self._widget_exists(existing_window):
            self.output_editor_window = None
            existing_window.destroy()

    def remove_output(self) -> None:
        ids = set(self.selected_output_ids())
        if not ids:
            return
        self.editing_outputs = [item for item in self.editing_outputs if item.id not in ids]
        self.clear_output_selection()
        self.refresh_outputs_tree()

    def log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        if threading.current_thread() is getattr(self, "ui_thread", None) and self._widget_exists(
            getattr(self, "log_text", None)
        ):
            self._append_log_line(line)
            self.root.update_idletasks()
            return
        self.log_queue.put(line)

    def _append_log_line(self, line: str) -> None:
        if not self._widget_exists(getattr(self, "log_text", None)):
            return
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log_line(line)
        except queue.Empty:
            pass

        with self.lock:
            active = len(self.active_jobs)
            pending = self.pending_jobs.qsize()
            state = (
                "準備中"
                if self.preparing
                else ("一時停止" if self.paused else ("実行中" if self.running else "待機中"))
            )
            jobs = list(self.all_jobs.values())

        self.status_var.set(f"{state} / 実行中 {active} / 待機 {pending}")
        self._sync_runtime_controls()
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
            self.log(
                f"{self.profile_display_label(profile)}: フォルダ設定が不足しています: "
                f"{profile_dir_label_text(missing_dirs)}"
            )
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
            self.log(f"{self.profile_display_label(profile)}: {len(self.files)} file(s) scanned.")

    def render_scan_rows(self, profile: EncodeProfile) -> None:
        self.progress_tree.delete(*self.progress_tree.get_children())
        for item in self.files:
            for variant in profile.outputs:
                if not variant.enabled or variant.id not in item.outputs:
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

    def refresh_encoder_capabilities(self, show_errors: bool = True) -> bool:
        try:
            self.encoder_capabilities = encoder_capabilities(self.paths.ffmpeg_path)
            self.encoder_smoke_cache = {}
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

    def prepare_ffmpeg_and_start(self, profile: EncodeProfile, specs: List[JobSpec]) -> None:
        with self.lock:
            busy = self.running or self.preparing
            if not busy:
                self.preparing = True
        if busy:
            messagebox.showwarning("実行中", "すでにジョブの準備または実行が進行中です。")
            return
        self._sync_runtime_controls()

        auto_download = not (self.paths.ffmpeg_path.exists() and self.paths.ffprobe_path.exists())
        if auto_download:
            messagebox.showinfo(
                "FFmpeg を準備します",
                "FFmpeg / FFprobe が見つからないため、自動でダウンロードして配置します。",
            )

        thread = threading.Thread(
            target=self._prepare_ffmpeg_worker,
            args=(profile, specs, auto_download),
            daemon=True,
        )
        thread.start()

    def _prepare_ffmpeg_worker(self, profile: EncodeProfile, specs: List[JobSpec], auto_download: bool) -> None:
        try:
            ensure_ffmpeg_available(self.paths, auto_download=auto_download, progress=self.log)
            if auto_download:
                self.log("FFmpeg / FFprobe is ready.")
            capabilities = encoder_capabilities(self.paths.ffmpeg_path)
        except Exception as exc:
            self.root.after(0, lambda error=exc: self._finish_ffmpeg_prepare_error(error))
            return
        self.root.after(0, lambda: self._finish_ffmpeg_prepare_success(profile, specs, capabilities))

    def _finish_ffmpeg_prepare_error(self, error: Exception) -> None:
        with self.lock:
            self.preparing = False
        self._sync_runtime_controls()
        messagebox.showerror("FFmpeg error", str(error))

    def _finish_ffmpeg_prepare_success(
        self,
        profile: EncodeProfile,
        specs: List[JobSpec],
        capabilities: Dict[str, Dict[str, object]],
    ) -> None:
        with self.lock:
            self.preparing = False
        self.encoder_capabilities = capabilities
        self.encoder_smoke_cache = {}
        count = len(self.encoder_capabilities)
        if count:
            self.log(f"FFmpeg encoder capabilities loaded: {count} encoder(s).")
        else:
            self.log("FFmpeg encoder capability scan returned no encoders.")
        self.update_output_encoder_controls()
        self._sync_runtime_controls()
        if not self.validate_encoder_capabilities_before_run(profile, specs):
            return
        self.start_specs(profile, specs, save=True)

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
                messagebox.showerror(
                    "入力エラー",
                    f"{variant.name}: CQ は CPU/NVENC のみで使用できます。QSV/AMF では VBR/ABR/CBR を選択してください。",
                )
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
            resource_id = (
                spec.assigned_resource_id
                if spec.assigned_resource_id in resource_ids
                else (resource_ids[0] if resource_ids else "")
            )
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

            target_resource_ids = resource_ids if encoder.endswith("_nvenc") else [resource_id]
            for target_resource_id in target_resource_ids:
                smoke_targets[(encoder, target_resource_id, split_mode)] = variant.name

        for (encoder, resource_id, split_mode), variant_name in smoke_targets.items():
            self.log(f"Smoke test: {encoder} on {resource_id or 'default'}")
            ok, detail = self.smoke_test_encoder_cached(encoder, resource_id, split_mode)
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

        self.prepare_ffmpeg_and_start(profile, specs)

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

        self.prepare_ffmpeg_and_start(profile, specs)

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

        self.log(f"{self.profile_display_label(profile)}: {len(specs)} job(s) added.")
        self.scheduler_thread = threading.Thread(target=self.scheduler_loop, args=(profile,), daemon=True)
        self.scheduler_thread.start()

    def create_runtime_job(self, spec: JobSpec, profile: EncodeProfile) -> RuntimeJob:
        self.job_counter += 1
        src = Path(spec.src)
        variant = variant_by_id(profile, spec.variant_id)
        resource_ids = variant_resource_ids(profile, variant)
        resource_id = (
            spec.assigned_resource_id
            if spec.assigned_resource_id in resource_ids
            else (resource_ids[0] if resource_ids else "")
        )
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
        limits = {resource.id: max(1, int(resource.concurrency_slots)) for resource in profile.hardware_resources}
        for variant in profile.outputs:
            for resource_id in variant_resource_ids(profile, variant):
                limits.setdefault(resource_id, 1)
        limits.setdefault(CPU_RESOURCE_ID, 1)
        return limits

    def resource_slot_weight(self, job: RuntimeJob, slot_limits: Dict[str, int]) -> int:
        resource_id = job.resource_id or CPU_RESOURCE_ID
        return max(1, slot_limits.get(resource_id, 1))

    def resource_detail_label(self, job: RuntimeJob) -> str:
        if not job.resource_id:
            return ""
        if not job.resource_slot_indexes:
            return job.resource_id
        indexes = job.resource_slot_indexes
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
        segment_seconds = max(60, variant_segment_minutes(job.profile, job.variant) * 60)
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
                    segment_root / segment_file_name(job.variant, index, src=src) for index, _range in enumerate(ranges)
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
                        audio_command = build_audio_command(
                            self.paths.ffmpeg_path, src, audio_out, job.profile, job.variant
                        )
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
        try:
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
                archive_dir = profile_archive_dir(profile, src)
                archive_dir.mkdir(parents=True, exist_ok=True)
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
