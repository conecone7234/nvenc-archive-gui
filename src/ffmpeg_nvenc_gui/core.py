from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

VIDEO_EXTS = {".mkv", ".m2ts", ".mp4", ".mov", ".ts"}
RESOLUTION_PRESETS: Dict[str, Optional[int]] = {
    "Original": None,
    "2160p": 2160,
    "1440p": 1440,
    "1080p": 1080,
    "720p": 720,
    "Custom": -1,
}
SAFE_CONTAINER_RE = re.compile(r"^[a-z0-9]{1,8}$")
FASTSTART_CONTAINERS = {"mp4", "m4v", "mov", "ismv"}
SEGMENT_SOURCE_STEM_LIMIT = 40
BACKEND_CPU = "cpu"
BACKEND_NVENC = "nvenc"
BACKEND_QSV = "qsv"
BACKEND_AMF = "amf"
BACKENDS = {BACKEND_CPU, BACKEND_NVENC, BACKEND_QSV, BACKEND_AMF}
CPU_RESOURCE_ID = "cpu:0"
DEFAULT_ENCODER_BY_BACKEND = {
    BACKEND_CPU: "libx264",
    BACKEND_NVENC: "hevc_nvenc",
    BACKEND_QSV: "hevc_qsv",
    BACKEND_AMF: "hevc_amf",
}
NVIDIA_NVENC_ENGINE_HINTS = [
    (re.compile(r"\brtx\s+5090\b", re.IGNORECASE), 3),
    (re.compile(r"\brtx\s+5080\b", re.IGNORECASE), 2),
    (re.compile(r"\brtx\s+5070\s*ti\b", re.IGNORECASE), 2),
    (re.compile(r"\brtx\s+4090\b", re.IGNORECASE), 2),
    (re.compile(r"\brtx\s+4080\b", re.IGNORECASE), 2),
    (re.compile(r"\brtx\s+4070\s*ti\b", re.IGNORECASE), 2),
]
AUDIO_CODEC_CHOICES = ["copy", "aac", "alac", "libmp3lame", "mp3", "opus", "vorbis", "flac"]
AUDIO_CONTAINERS_BY_CODEC = {
    "copy": ["mka", "mkv"],
    "aac": ["m4a", "mp4", "mov"],
    "alac": ["m4a", "mp4", "mov"],
    "libmp3lame": ["mp3"],
    "mp3": ["mp3"],
    "opus": ["mka", "mkv", "webm"],
    "vorbis": ["mka", "mkv", "webm"],
    "flac": ["mka", "mkv"],
}
SOURCE_TEMPLATE_TOKENS = ("${filename}", "{source}")
STALE_AUTO_RESOURCE_ERRORS = {
    "Manual resource; availability is checked before encoding.",
}
MANUAL_RESOURCE_ERROR_PREFIX = "Manual fallback resource;"
DEPRECATED_PROFILE_FIELDS = {
    "max_parallel_jobs",
    "use_gpu",
    "gpu_index",
    "gpu_name",
    "codec",
    "cpu_codec",
    "preset",
    "cpu_preset",
    "tune",
    "cpu_tune",
    "rate_mode",
    "cq_value",
    "bitrate",
    "maxrate",
    "bufsize",
    "pix_fmt",
    "scale_flags",
}


def dataclass_values(cls: Any, data: object) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


@dataclass
class AppPaths:
    base_dir: Path
    ffmpeg_path: Path
    ffprobe_path: Path
    tmp_dir: Path
    log_dir: Path
    state_file: Path
    config_file: Path
    download_dir: Path


@dataclass
class GpuInfo:
    index: int
    name: str
    vendor: str = "nvidia"
    encoder_engines: Optional[int] = None
    detection_error: str = ""

    def __post_init__(self) -> None:
        self.index = normalize_int(self.index, minimum=0, default=0)
        self.name = str(self.name or "").strip()
        self.vendor = str(self.vendor or "nvidia").strip().lower()
        self.encoder_engines = normalize_optional_int(self.encoder_engines, minimum=1)
        self.detection_error = str(self.detection_error or "").strip()


@dataclass
class HardwareResource:
    id: str
    label: str
    kind: str
    backend: str
    vendor: str = ""
    index: int = 0
    concurrency_slots: int = 1
    detected_encoder_engines: Optional[int] = None
    detection_error: str = ""

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.label = str(self.label or self.id).strip()
        self.kind = str(self.kind or "").strip().lower()
        self.backend = normalize_backend(self.backend)
        self.vendor = str(self.vendor or "").strip().lower()
        self.index = normalize_int(self.index, minimum=0, default=0)
        self.concurrency_slots = normalize_int(self.concurrency_slots, minimum=1, default=1)
        self.detected_encoder_engines = normalize_optional_int(self.detected_encoder_engines, minimum=1)
        self.detection_error = str(self.detection_error or "").strip()


@dataclass
class OutputVariant:
    id: str
    name: str
    folder_name: str
    height: Optional[int] = None
    container: str = "mp4"
    enabled: bool = True
    filename_template: str = "{source}"
    input_dir: str = ""
    output_dir: str = ""
    segment_minutes: Optional[int] = None
    use_gpu: Optional[bool] = None
    gpu_index: Optional[int] = None
    gpu_name: str = ""
    codec: str = ""
    cpu_codec: str = ""
    preset: str = ""
    cpu_preset: str = ""
    cpu_tune: str = ""
    tune: str = ""
    rate_mode: str = ""
    cq_value: Optional[int] = None
    bitrate: str = ""
    maxrate: str = ""
    bufsize: str = ""
    pix_fmt: str = ""
    scale_flags: str = ""
    audio_codec: str = ""
    audio_bitrate: str = ""
    audio_container: str = ""
    backend: str = ""
    ffmpeg_encoder: str = ""
    resource_ids: List[str] = field(default_factory=list)
    concurrency_policy: str = "resource_slots"
    split_encode_mode: str = ""
    extra_input_args: str = ""
    extra_video_args: str = ""
    extra_audio_args: str = ""
    extra_output_args: str = ""
    extra_concat_args: str = ""
    extra_mux_args: str = ""

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.name = str(self.name or "").strip()
        self.folder_name = str(self.folder_name or "").strip()
        if not self.id:
            self.id = new_id("variant")
        if self.height in ("", None):
            self.height = None
        else:
            try:
                self.height = int(self.height)
            except (TypeError, ValueError):
                self.height = None
            if self.height is not None and self.height < 1:
                self.height = None
        if isinstance(self.enabled, str):
            self.enabled = self.enabled.strip().lower() not in {"0", "false", "no", "off"}
        else:
            self.enabled = bool(self.enabled)
        self.folder_name = safe_folder_name(self.folder_name or self.name or self.id)
        self.container = normalize_container_extension(self.container)
        self.filename_template = str(self.filename_template or "{source}").strip() or "{source}"
        self.input_dir = str(self.input_dir or "").strip()
        self.output_dir = str(self.output_dir or "").strip()
        self.segment_minutes = normalize_optional_int(self.segment_minutes, minimum=1)
        self.use_gpu = normalize_optional_bool(self.use_gpu)
        self.gpu_index = normalize_optional_int(self.gpu_index, minimum=0)
        self.gpu_name = str(self.gpu_name or "").strip()
        self.codec = str(self.codec or "").strip()
        self.cpu_codec = str(self.cpu_codec or "").strip()
        self.preset = str(self.preset or "").strip()
        self.cpu_preset = str(self.cpu_preset or "").strip()
        self.cpu_tune = str(self.cpu_tune or "").strip()
        self.tune = str(self.tune or "").strip()
        self.rate_mode = str(self.rate_mode or "").strip().upper()
        self.cq_value = normalize_optional_int(self.cq_value)
        self.bitrate = str(self.bitrate or "").strip()
        self.maxrate = str(self.maxrate or "").strip()
        self.bufsize = str(self.bufsize or "").strip()
        self.pix_fmt = str(self.pix_fmt or "").strip()
        self.scale_flags = str(self.scale_flags or "").strip()
        self.audio_codec = normalize_audio_codec(self.audio_codec)
        self.audio_bitrate = str(self.audio_bitrate or "").strip()
        self.audio_container = normalize_audio_container_for_codec(self.audio_codec, self.audio_container)
        self.ffmpeg_encoder = str(self.ffmpeg_encoder or "").strip()
        inherit_profile_device = (
            not self.backend and not self.ffmpeg_encoder and self.use_gpu is None and not self.resource_ids
        )
        if inherit_profile_device:
            self.backend = ""
        elif not self.backend:
            if self.ffmpeg_encoder:
                self.backend = backend_from_encoder(self.ffmpeg_encoder)
            elif self.use_gpu is True:
                self.backend = BACKEND_NVENC
            else:
                self.backend = BACKEND_CPU
        if self.backend:
            self.backend = normalize_backend(self.backend)
        if not self.ffmpeg_encoder:
            if not self.backend:
                self.ffmpeg_encoder = ""
            elif self.backend == BACKEND_NVENC:
                self.ffmpeg_encoder = self.codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_NVENC]
            elif self.backend == BACKEND_CPU:
                self.ffmpeg_encoder = self.cpu_codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU]
            else:
                self.ffmpeg_encoder = DEFAULT_ENCODER_BY_BACKEND[self.backend]
        if self.backend == BACKEND_NVENC and not self.codec:
            self.codec = self.ffmpeg_encoder
        if self.backend == BACKEND_CPU and not self.cpu_codec:
            self.cpu_codec = self.ffmpeg_encoder
        if not self.resource_ids and self.backend:
            if self.backend == BACKEND_CPU:
                self.resource_ids = [CPU_RESOURCE_ID]
            elif self.backend == BACKEND_NVENC:
                self.resource_ids = [resource_id_for_backend(BACKEND_NVENC, self.gpu_index or 0)]
            else:
                self.resource_ids = [resource_id_for_backend(self.backend, 0)]
        self.resource_ids = normalize_resource_id_list(self.resource_ids)
        self.concurrency_policy = str(self.concurrency_policy or "resource_slots").strip() or "resource_slots"
        self.split_encode_mode = str(self.split_encode_mode or "").strip().lower()
        self.extra_input_args = str(self.extra_input_args or "").strip()
        self.extra_video_args = str(self.extra_video_args or "").strip()
        self.extra_audio_args = str(self.extra_audio_args or "").strip()
        self.extra_output_args = str(self.extra_output_args or "").strip()
        self.extra_concat_args = str(self.extra_concat_args or "").strip()
        self.extra_mux_args = str(self.extra_mux_args or "").strip()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OutputVariant":
        base = {
            "id": "",
            "name": "",
            "folder_name": "",
            "height": None,
            "container": "mp4",
            "enabled": True,
            "input_dir": "",
            "output_dir": "",
            "segment_minutes": None,
        }
        base.update(dataclass_values(OutputVariant, data))
        return OutputVariant(**base)


EncodeSet = OutputVariant


@dataclass
class EncodeProfile:
    id: str
    input_dir: str
    output_dir: str
    archive_dir: str
    max_parallel_jobs: int = 2
    segment_minutes: int = 10
    use_gpu: bool = True
    gpu_index: int = 0
    gpu_name: str = ""
    codec: str = "hevc_nvenc"
    cpu_codec: str = "libx264"
    preset: str = "p7"
    cpu_preset: str = "medium"
    tune: str = "hq"
    cpu_tune: str = "none"
    rate_mode: str = "CQ"
    cq_value: int = 18
    bitrate: str = "25000k"
    maxrate: str = "40000k"
    bufsize: str = "80000k"
    pix_fmt: str = "nv12"
    scale_flags: str = "lanczos+accurate_rnd"
    resource_ids: List[str] = field(default_factory=list)
    hardware_resources: List[HardwareResource] = field(default_factory=list)
    outputs: List[OutputVariant] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip() or new_id("profile")
        self.input_dir = str(self.input_dir or "").strip()
        self.output_dir = str(self.output_dir or "").strip()
        self.archive_dir = str(self.archive_dir or "").strip()
        self.gpu_name = str(self.gpu_name or "").strip()
        self.codec = str(self.codec or "hevc_nvenc").strip()
        self.cpu_codec = str(self.cpu_codec or "libx264").strip()
        self.preset = str(self.preset or "p7").strip()
        self.cpu_preset = str(self.cpu_preset or "medium").strip()
        self.tune = str(self.tune or "hq").strip()
        self.cpu_tune = str(self.cpu_tune or "none").strip()
        self.rate_mode = str(self.rate_mode or "CQ").strip().upper()
        self.bitrate = str(self.bitrate or "").strip()
        self.maxrate = str(self.maxrate or "").strip()
        self.bufsize = str(self.bufsize or "").strip()
        self.pix_fmt = str(self.pix_fmt or "nv12").strip()
        self.scale_flags = str(self.scale_flags or "lanczos+accurate_rnd").strip()

        try:
            self.max_parallel_jobs = max(1, int(self.max_parallel_jobs))
        except (TypeError, ValueError):
            self.max_parallel_jobs = 1
        try:
            self.segment_minutes = max(1, int(self.segment_minutes))
        except (TypeError, ValueError):
            self.segment_minutes = 10
        try:
            self.gpu_index = max(0, int(self.gpu_index))
        except (TypeError, ValueError):
            self.gpu_index = 0
        try:
            self.cq_value = int(self.cq_value)
        except (TypeError, ValueError):
            self.cq_value = 18

        if isinstance(self.use_gpu, str):
            self.use_gpu = self.use_gpu.strip().lower() not in {"0", "false", "no", "off"}
        else:
            self.use_gpu = bool(self.use_gpu)

        self.resource_ids = normalize_resource_id_list(self.resource_ids)
        raw_resources = self.hardware_resources if isinstance(self.hardware_resources, list) else []
        resources: List[HardwareResource] = []
        seen_resource_ids: set[str] = set()
        for resource in raw_resources:
            if isinstance(resource, HardwareResource):
                item = resource
            elif isinstance(resource, dict):
                values = {
                    "id": "",
                    "label": "",
                    "kind": "",
                    "backend": "",
                }
                values.update(dataclass_values(HardwareResource, resource))
                item = HardwareResource(**values)
            else:
                continue
            if not item.id or item.id in seen_resource_ids:
                continue
            seen_resource_ids.add(item.id)
            resources.append(item)
        self.hardware_resources = resources

        raw_outputs = self.outputs if isinstance(self.outputs, list) else []
        normalized_outputs: List[OutputVariant] = []
        seen_output_ids: set[str] = set()
        for output in raw_outputs:
            if isinstance(output, OutputVariant):
                variant = output
            else:
                variant = OutputVariant.from_dict(output)
            while variant.id in seen_output_ids:
                variant.id = new_id("variant")
            seen_output_ids.add(variant.id)
            normalized_outputs.append(variant)
        self.outputs = normalized_outputs

    @staticmethod
    def from_dict(
        data: Dict[str, Any],
        paths: Optional[AppPaths] = None,
        gpus: Optional[List[GpuInfo]] = None,
    ) -> "EncodeProfile":
        base_paths = paths or build_paths()
        default_gpus = [] if gpus is None else gpus
        base = asdict(default_profile(base_paths, default_gpus))
        explicit_values = dataclass_values(EncodeProfile, data)
        base.update(explicit_values)
        raw_outputs = base.get("outputs")
        if not isinstance(raw_outputs, list):
            raw_outputs = []
        base["outputs"] = [OutputVariant.from_dict(x) for x in raw_outputs]
        if not base["outputs"]:
            base["outputs"] = default_outputs()
        profile = normalize_profile_gpu(EncodeProfile(**base), default_gpus)
        if not profile.use_gpu:
            cpu_defaults = default_profile(base_paths, [])
            for key in ("max_parallel_jobs", "cq_value", "bitrate", "maxrate", "bufsize", "cpu_preset", "cpu_tune"):
                if key not in explicit_values:
                    setattr(profile, key, getattr(cpu_defaults, key))
        return profile


@dataclass
class JobSpec:
    src: str
    profile_id: str
    variant_id: str
    assigned_resource_id: str = ""
    assigned_slot: int = 0

    def __post_init__(self) -> None:
        self.src = str(self.src or "")
        self.profile_id = str(self.profile_id or "")
        self.variant_id = str(self.variant_id or "")
        self.assigned_resource_id = str(self.assigned_resource_id or "").strip().lower()
        self.assigned_slot = normalize_int(self.assigned_slot, minimum=0, default=0)


@dataclass
class FileStatus:
    path: Path
    outputs: Dict[str, bool]

    def label(self, profile: EncodeProfile) -> str:
        enabled = [variant for variant in profile.outputs if variant.enabled and variant.id in self.outputs]
        if not enabled:
            enabled = [variant for variant in profile.outputs if variant.enabled]
        done = sum(1 for variant in enabled if self.outputs.get(variant.id, False))
        total = len(enabled)
        if total == 0:
            return "出力なし"
        if done == total:
            return "完了"
        if done == 0:
            return "未処理"
        return f"残り {total - done}/{total}"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def normalize_optional_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        return text not in {"0", "false", "no", "off", "none", "auto"}
    return bool(value)


def normalize_int(value: object, minimum: Optional[int] = None, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    return number


def normalize_optional_int(value: object, minimum: Optional[int] = None) -> Optional[int]:
    if value in ("", None):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None:
        number = max(minimum, number)
    return number


def normalize_backend(value: object) -> str:
    backend = str(value or "").strip().lower()
    return backend if backend in BACKENDS else BACKEND_CPU


def backend_from_encoder(encoder: object) -> str:
    value = str(encoder or "").strip().lower()
    if value.endswith("_nvenc"):
        return BACKEND_NVENC
    if value.endswith("_qsv"):
        return BACKEND_QSV
    if value.endswith("_amf"):
        return BACKEND_AMF
    return BACKEND_CPU


def resource_id_for_backend(backend: str, index: int = 0) -> str:
    backend = normalize_backend(backend)
    if backend == BACKEND_CPU:
        return f"cpu:{max(0, int(index))}"
    vendor = {
        BACKEND_NVENC: "nvidia",
        BACKEND_QSV: "intel",
        BACKEND_AMF: "amd",
    }.get(backend, backend)
    return f"{vendor}:{max(0, int(index))}"


def resource_backend(resource_id: str) -> str:
    prefix = str(resource_id or "").split(":", 1)[0].lower()
    if prefix == "cpu":
        return BACKEND_CPU
    if prefix == "nvidia":
        return BACKEND_NVENC
    if prefix == "intel":
        return BACKEND_QSV
    if prefix == "amd":
        return BACKEND_AMF
    return BACKEND_CPU


def resource_index(resource_id: str) -> int:
    _prefix, sep, suffix = str(resource_id or "").partition(":")
    if not sep:
        return 0
    return normalize_int(suffix, minimum=0, default=0)


def normalize_resource_id_list(value: object) -> List[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip().lower()
        if not text or ":" not in text:
            continue
        prefix, _, suffix = text.partition(":")
        if prefix not in {"cpu", "nvidia", "intel", "amd"}:
            continue
        resource_id = f"{prefix}:{normalize_int(suffix, minimum=0, default=0)}"
        if resource_id not in normalized:
            normalized.append(resource_id)
    return normalized


def normalize_audio_codec(value: object) -> str:
    codec = str(value or "copy").strip().lower()
    return codec if codec in AUDIO_CONTAINERS_BY_CODEC else "copy"


def audio_containers_for_codec(codec: object) -> List[str]:
    return list(AUDIO_CONTAINERS_BY_CODEC.get(normalize_audio_codec(codec), AUDIO_CONTAINERS_BY_CODEC["copy"]))


def normalize_audio_container_for_codec(codec: object, container: object) -> str:
    containers = audio_containers_for_codec(codec)
    value = normalize_container_extension(container, default="")
    return value if value in containers else containers[0]


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def build_paths(base_dir: Optional[Path] = None) -> AppPaths:
    base = (base_dir or get_base_dir()).resolve()
    tmp = base / "tmp"
    return AppPaths(
        base_dir=base,
        ffmpeg_path=base / "ffmpeg" / "ffmpeg.exe",
        ffprobe_path=base / "ffmpeg" / "ffprobe.exe",
        tmp_dir=tmp,
        log_dir=tmp / "logs",
        state_file=tmp / "encoder_state.json",
        config_file=base / "profiles.json",
        download_dir=tmp / "downloads",
    )


def ensure_dirs(paths: AppPaths) -> None:
    paths.tmp_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.download_dir.mkdir(parents=True, exist_ok=True)
    paths.ffmpeg_path.parent.mkdir(parents=True, exist_ok=True)


def missing_profile_dirs(profile: EncodeProfile) -> List[str]:
    required = [
        ("input_dir", profile.input_dir),
        ("output_dir", profile.output_dir),
        ("archive_dir", profile.archive_dir),
    ]
    return [name for name, value in required if not str(value or "").strip()]


def validate_profile_dirs(profile: EncodeProfile) -> None:
    missing = missing_profile_dirs(profile)
    if missing:
        raise ValueError(f"profile requires: {', '.join(missing)}")


def ensure_profile_dirs(profile: EncodeProfile) -> None:
    validate_profile_dirs(profile)
    input_dir = profile_input_dir(profile)

    input_dir.mkdir(parents=True, exist_ok=True)
    if not has_source_template(profile.output_dir):
        profile_output_dir(profile).mkdir(parents=True, exist_ok=True)
    if not has_source_template(profile.archive_dir):
        profile_archive_dir(profile).mkdir(parents=True, exist_ok=True)
    for variant in profile.outputs:
        if variant.enabled:
            variant_input_dir(profile, variant).mkdir(parents=True, exist_ok=True)
            output_template = variant.output_dir or profile.output_dir
            if not has_source_template(output_template) and not has_source_template(variant.folder_name):
                (variant_output_dir(profile, variant) / expand_folder_template(variant.folder_name, None)).mkdir(
                    parents=True,
                    exist_ok=True,
                )


def safe_folder_name(text: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", text.strip())
    value = re.sub(r"\s+", "-", value)
    value = value.strip(".- ")
    return value or "output"


def safe_file_stem(text: str, default: str = "video") -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", str(text or "").strip())
    value = re.sub(r"\s+", " ", value)
    value = value.strip(".- ")
    return value or default


def normalize_container_extension(value: object, default: str = "mp4") -> str:
    candidate = str(value or "").strip().lower().lstrip(".")
    if SAFE_CONTAINER_RE.fullmatch(candidate):
        return candidate
    return default


def has_source_template(value: object) -> bool:
    text = str(value or "")
    return any(token in text for token in SOURCE_TEMPLATE_TOKENS)


def source_template_replacements(src: Path) -> Dict[str, str]:
    return {
        "${filename}": safe_folder_name(src.stem),
        "{source}": safe_folder_name(src.stem),
    }


def expand_path_template(value: object, src: Optional[Path]) -> str:
    text = str(value or "").strip()
    if src is None:
        return text
    for token, replacement in source_template_replacements(src).items():
        text = text.replace(token, replacement)
    return text


def expand_folder_template(value: object, src: Optional[Path]) -> str:
    expanded = expand_path_template(value, src)
    return safe_folder_name(expanded)


def parse_ffmpeg_args(value: object) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        lexer = shlex.shlex(text.replace("\r", " ").replace("\n", " "), posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.escape = ""
        return list(lexer)
    except ValueError as exc:
        raise ValueError(f"Invalid FFmpeg options: {exc}") from exc


def has_ffmpeg_option(args: Iterable[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def faststart_args_for_container(value: object) -> List[str]:
    container = normalize_container_extension(value)
    if container in FASTSTART_CONTAINERS:
        return ["-movflags", "+faststart"]
    return []


def faststart_args_unless_overridden(value: object, extra_args: Iterable[str]) -> List[str]:
    if has_ffmpeg_option(extra_args, "-movflags"):
        return []
    return faststart_args_for_container(value)


def default_outputs() -> List[OutputVariant]:
    return [
        OutputVariant(
            id="master_2160p",
            name="UP Convert 4K",
            folder_name="up-convert-4k",
            height=2160,
            container="mp4",
        ),
        OutputVariant(
            id="reference_1080p",
            name="ReEncode Original Pixel",
            folder_name="reencode-original-pixel",
            height=None,
            container="mp4",
        ),
    ]


def detect_nvenc_engine_count(_gpu_index: int = 0) -> Tuple[Optional[int], str]:
    if os.name != "nt":
        return None, "NVENC engine probing is only attempted on Windows."
    try:
        ctypes.WinDLL("nvEncodeAPI64.dll")
    except Exception as exc:
        return None, f"nvEncodeAPI64.dll unavailable: {exc}"
    # Opening an NVENC session requires a CUDA/D3D device context. The app keeps this
    # probe optional and falls back to user-tunable slots when no helper context exists.
    return None, "nvEncodeAPI64.dll loaded; encode-session caps helper unavailable, using manual slots."


def nvenc_engine_hint_from_name(name: str) -> Optional[int]:
    normalized = re.sub(r"[\s_-]+", " ", str(name or "")).strip()
    for pattern, engines in NVIDIA_NVENC_ENGINE_HINTS:
        if pattern.search(normalized):
            return engines
    return None


def fallback_cpu_resource() -> HardwareResource:
    return HardwareResource(
        id=CPU_RESOURCE_ID,
        label="CPU",
        kind="cpu",
        backend=BACKEND_CPU,
        vendor="cpu",
        index=0,
        concurrency_slots=1,
    )


def cpu_resource_from_wmi_item(item: Dict[str, object], fallback_index: int) -> Optional[HardwareResource]:
    name = str(item.get("Name") or "").strip()
    device_id = str(item.get("DeviceID") or "").strip()
    socket = str(item.get("SocketDesignation") or "").strip()
    index = fallback_index
    match = re.search(r"(\d+)", device_id)
    if match:
        index = normalize_int(match.group(1), minimum=0, default=fallback_index)
    if not name and not socket and not device_id:
        return None
    model = name or "CPU"
    socket_label = socket or device_id
    label = f"CPU {index}: {model}"
    if socket_label:
        label = f"{label} / {socket_label}"
    return HardwareResource(
        id=resource_id_for_backend(BACKEND_CPU, index),
        label=label,
        kind="cpu",
        backend=BACKEND_CPU,
        vendor="cpu",
        index=index,
        concurrency_slots=1,
    )


def cpu_resources_from_wmi_data(data: object) -> List[HardwareResource]:
    items = data if isinstance(data, list) else [data]
    resources: List[HardwareResource] = []
    seen: set[str] = set()
    for fallback_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        resource = cpu_resource_from_wmi_item(item, fallback_index)
        if resource is None or resource.id in seen:
            continue
        resources.append(resource)
        seen.add(resource.id)
    return resources or [fallback_cpu_resource()]


@lru_cache(maxsize=1)
def _detect_cpu_resources_cached(timeout: int = 5) -> Tuple[HardwareResource, ...]:
    if os.name != "nt":
        return (fallback_cpu_resource(),)
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_Processor | Select-Object DeviceID,Name,SocketDesignation | ConvertTo-Json -Compress",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except Exception:
        return (fallback_cpu_resource(),)
    if result.returncode != 0:
        return (fallback_cpu_resource(),)
    try:
        data = json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        return (fallback_cpu_resource(),)
    return tuple(cpu_resources_from_wmi_data(data))


def detect_cpu_resources(timeout: int = 5) -> List[HardwareResource]:
    return list(_detect_cpu_resources_cached(timeout))


def clear_cpu_resource_cache() -> None:
    _detect_cpu_resources_cached.cache_clear()


def hardware_resources_from_gpus(gpus: Optional[List[GpuInfo]]) -> List[HardwareResource]:
    resources = detect_cpu_resources()
    for gpu in gpus or []:
        backend = {
            "nvidia": BACKEND_NVENC,
            "intel": BACKEND_QSV,
            "amd": BACKEND_AMF,
        }.get(gpu.vendor, BACKEND_NVENC)
        engine_count = gpu.encoder_engines
        detection_error = gpu.detection_error
        if backend == BACKEND_NVENC and engine_count is None:
            engine_count, detection_error = detect_nvenc_engine_count(gpu.index)
            if engine_count is None:
                hinted_count = nvenc_engine_hint_from_name(gpu.name)
                if hinted_count is not None:
                    engine_count = hinted_count
                    detection_error = (
                        f"{detection_error} Model hint selected {hinted_count} NVENC engine(s); "
                        "encoder availability is still checked before encoding."
                    ).strip()
        slots = engine_count or 1
        resources.append(
            HardwareResource(
                id=resource_id_for_backend(backend, gpu.index),
                label=f"GPU {gpu.index}: {gpu.name}",
                kind="gpu",
                backend=backend,
                vendor=gpu.vendor,
                index=gpu.index,
                concurrency_slots=slots,
                detected_encoder_engines=engine_count,
                detection_error=detection_error,
            )
        )
    return resources


def normalize_hardware_resources(
    resources: Iterable[HardwareResource],
    gpus: Optional[List[GpuInfo]],
) -> List[HardwareResource]:
    by_id = {resource.id: resource for resource in hardware_resources_from_gpus(gpus)}
    for resource in resources:
        if not resource.id:
            continue
        if resource.id in by_id:
            detected = by_id[resource.id]
            detected.concurrency_slots = resource.concurrency_slots
            by_id[resource.id] = detected
            continue
        if resource.id not in by_id and resource.detection_error in STALE_AUTO_RESOURCE_ERRORS:
            continue
        if (
            resource.id not in by_id
            and resource_backend(resource.id) == BACKEND_NVENC
            and not resource.detection_error.startswith(MANUAL_RESOURCE_ERROR_PREFIX)
        ):
            continue
        by_id[resource.id] = resource
    if CPU_RESOURCE_ID not in by_id:
        by_id[CPU_RESOURCE_ID] = fallback_cpu_resource()
    return sorted(by_id.values(), key=lambda item: (item.kind != "cpu", item.vendor, item.index, item.id))


def detect_nvidia_gpus(timeout: int = 5) -> List[GpuInfo]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    gpus: List[GpuInfo] = []
    for line in result.stdout.splitlines():
        if not line.strip() or "," not in line:
            continue
        index_text, name = [part.strip() for part in line.split(",", 1)]
        if not index_text.isdigit() or not name:
            continue
        gpus.append(GpuInfo(index=int(index_text), name=name))
    return gpus


def gpu_vendor_from_wmi_item(item: Dict[str, object]) -> str:
    haystack = " ".join(str(item.get(key) or "") for key in ("Name", "PNPDeviceID", "AdapterCompatibility")).lower()
    if "10de" in haystack or "nvidia" in haystack:
        return "nvidia"
    if "8086" in haystack or "intel" in haystack:
        return "intel"
    if "1002" in haystack or "amd" in haystack or "advanced micro" in haystack or "radeon" in haystack:
        return "amd"
    return ""


def gpus_from_wmi_data(data: object, existing: Optional[Iterable[GpuInfo]] = None) -> List[GpuInfo]:
    items = data if isinstance(data, list) else [data]
    counts: Dict[str, int] = {"nvidia": 0, "intel": 0, "amd": 0}
    existing_keys = {
        (gpu.vendor, re.sub(r"\s+", " ", gpu.name).strip().lower()) for gpu in (existing or []) if gpu.name
    }
    gpus: List[GpuInfo] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        vendor = gpu_vendor_from_wmi_item(item)
        if not name or vendor not in counts:
            continue
        key = (vendor, re.sub(r"\s+", " ", name).strip().lower())
        if key in existing_keys:
            counts[vendor] += 1
            continue
        index = counts[vendor]
        counts[vendor] += 1
        gpus.append(GpuInfo(index=index, name=name, vendor=vendor))
    return gpus


def detect_wmi_gpus(timeout: int = 5, existing: Optional[Iterable[GpuInfo]] = None) -> List[GpuInfo]:
    if os.name != "nt":
        return []
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,PNPDeviceID,AdapterCompatibility | ConvertTo-Json -Compress"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        return []
    return gpus_from_wmi_data(data, existing)


def detect_gpus(timeout: int = 5) -> List[GpuInfo]:
    nvidia = detect_nvidia_gpus(timeout)
    return nvidia + detect_wmi_gpus(timeout, existing=nvidia)


def normalize_profile_gpu(profile: EncodeProfile, gpus: Optional[List[GpuInfo]]) -> EncodeProfile:
    profile.hardware_resources = normalize_hardware_resources(profile.hardware_resources, gpus)
    resource_by_id = {resource.id: resource for resource in profile.hardware_resources}
    if not profile.resource_ids:
        if profile.use_gpu:
            profile.resource_ids = [resource_id_for_backend(BACKEND_NVENC, profile.gpu_index)]
        else:
            profile.resource_ids = [CPU_RESOURCE_ID]
    profile.resource_ids = [resource_id for resource_id in profile.resource_ids if resource_id in resource_by_id]
    if not profile.resource_ids:
        profile.resource_ids = [CPU_RESOURCE_ID]

    gpu_by_index = {gpu.index: gpu for gpu in gpus or []}
    if profile.use_gpu:
        gpu = gpu_by_index.get(profile.gpu_index)
        if gpu is not None:
            profile.gpu_name = gpu.name
        else:
            profile.use_gpu = False
            profile.gpu_index = 0
            profile.gpu_name = ""

    for variant in profile.outputs:
        if not variant.backend:
            if profile.use_gpu:
                variant.backend = BACKEND_NVENC
                variant.ffmpeg_encoder = variant.codec or profile.codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_NVENC]
                variant.resource_ids = [resource_id_for_backend(BACKEND_NVENC, profile.gpu_index)]
            else:
                variant.backend = BACKEND_CPU
                variant.ffmpeg_encoder = (
                    variant.cpu_codec or profile.cpu_codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU]
                )
                variant.resource_ids = [CPU_RESOURCE_ID]
        variant.backend = normalize_backend(variant.backend)
        if not variant.ffmpeg_encoder:
            variant.ffmpeg_encoder = DEFAULT_ENCODER_BY_BACKEND[variant.backend]
        valid_resources = [
            resource_id
            for resource_id in variant.resource_ids
            if resource_id in resource_by_id and resource_backend(resource_id) == variant.backend
        ]
        if not valid_resources:
            if variant.use_gpu is True:
                candidate = resource_id_for_backend(
                    BACKEND_NVENC, variant.gpu_index if variant.gpu_index is not None else profile.gpu_index
                )
                if candidate in resource_by_id:
                    valid_resources = [candidate]
            elif variant.backend == BACKEND_CPU and CPU_RESOURCE_ID in resource_by_id:
                valid_resources = [CPU_RESOURCE_ID]
        if not valid_resources:
            variant.backend = BACKEND_CPU
            variant.ffmpeg_encoder = variant.cpu_codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU]
            variant.use_gpu = False
            variant.gpu_index = 0
            variant.gpu_name = ""
            variant.resource_ids = [CPU_RESOURCE_ID]
        else:
            variant.resource_ids = valid_resources
            variant.use_gpu = variant.backend != BACKEND_CPU
            if variant.backend == BACKEND_NVENC:
                gpu_index = resource_index(valid_resources[0])
                gpu = gpu_by_index.get(gpu_index)
                variant.gpu_index = gpu_index
                variant.gpu_name = gpu.name if gpu else ""
                if not variant.codec:
                    variant.codec = variant.ffmpeg_encoder or DEFAULT_ENCODER_BY_BACKEND[BACKEND_NVENC]
            elif variant.backend == BACKEND_CPU:
                variant.gpu_index = 0
                variant.gpu_name = ""
                if not variant.cpu_codec:
                    variant.cpu_codec = variant.ffmpeg_encoder or DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU]
    return profile


def default_profile(paths: AppPaths, gpus: Optional[List[GpuInfo]] = None) -> EncodeProfile:
    detected = detect_gpus() if gpus is None else gpus
    gpu = next((item for item in detected if item.vendor == "nvidia"), None)
    hardware_resources = hardware_resources_from_gpus(detected)
    resource_ids = [
        resource_id_for_backend(
            {
                "nvidia": BACKEND_NVENC,
                "intel": BACKEND_QSV,
                "amd": BACKEND_AMF,
            }.get(gpu_info.vendor, BACKEND_NVENC),
            gpu_info.index,
        )
        for gpu_info in detected
    ] or [CPU_RESOURCE_ID]
    output_backend = resource_backend(resource_ids[0])
    output_resource_ids = [
        resource_id for resource_id in resource_ids if resource_backend(resource_id) == output_backend
    ]
    outputs = default_outputs()
    for output in outputs:
        output.backend = output_backend
        output.ffmpeg_encoder = DEFAULT_ENCODER_BY_BACKEND[output_backend]
        output.resource_ids = list(output_resource_ids)
        output.use_gpu = output_backend != BACKEND_CPU
        if output_backend == BACKEND_NVENC:
            output.codec = output.ffmpeg_encoder
            output.gpu_index = resource_index(output_resource_ids[0]) if output_resource_ids else 0
        elif output_backend == BACKEND_CPU:
            output.cpu_codec = output.ffmpeg_encoder
    return EncodeProfile(
        id="default",
        input_dir=str(paths.base_dir / "Incoming"),
        output_dir=str(paths.base_dir / "output" / "{source}"),
        archive_dir=str(paths.base_dir / "output" / "{source}"),
        max_parallel_jobs=2 if gpu else 1,
        segment_minutes=10,
        use_gpu=gpu is not None,
        gpu_index=gpu.index if gpu else 0,
        gpu_name=gpu.name if gpu else "",
        codec="hevc_nvenc",
        cpu_codec="libx264",
        cpu_preset="medium",
        cpu_tune="none",
        cq_value=15 if gpu else 23,
        bitrate="25000k" if gpu else "8000k",
        maxrate="40000k" if gpu else "12000k",
        bufsize="80000k" if gpu else "24000k",
        resource_ids=resource_ids,
        hardware_resources=hardware_resources,
        outputs=outputs,
    )


def read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def load_profiles(paths: AppPaths, gpus: Optional[List[GpuInfo]] = None) -> List[EncodeProfile]:
    if not paths.config_file.exists():
        return [default_profile(paths, gpus)]

    data = read_json_object(paths.config_file)
    if data is None:
        return [default_profile(paths, gpus)]
    raw_profiles = data.get("profiles")
    if not isinstance(raw_profiles, list):
        return [default_profile(paths, gpus)]
    profiles = [EncodeProfile.from_dict(item, paths, gpus) for item in raw_profiles if isinstance(item, dict)]
    return profiles or [default_profile(paths, gpus)]


def save_profiles(paths: AppPaths, profiles: List[EncodeProfile]) -> None:
    ensure_dirs(paths)
    data = {
        "version": 2,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profiles": [profile_to_dict(profile) for profile in profiles],
    }
    tmp_file = paths.config_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(paths.config_file)


def output_variant_to_dict(variant: OutputVariant) -> Dict[str, Any]:
    data = asdict(variant)
    if variant.backend and variant.ffmpeg_encoder:
        data.pop("use_gpu", None)
        data.pop("gpu_index", None)
        data.pop("gpu_name", None)
    return data


def profile_to_dict(profile: EncodeProfile) -> Dict[str, Any]:
    data = asdict(profile)
    for key in DEPRECATED_PROFILE_FIELDS:
        data.pop(key, None)
    data["outputs"] = [output_variant_to_dict(variant) for variant in profile.outputs]
    data["hardware_resources"] = [asdict(resource) for resource in profile.hardware_resources]
    return data


def variant_by_id(profile: EncodeProfile, variant_id: str) -> OutputVariant:
    for variant in profile.outputs:
        if variant.id == variant_id:
            return variant
    raise ValueError(f"Unknown output variant: {variant_id}")


def profile_input_dir(profile: EncodeProfile) -> Path:
    return Path(profile.input_dir).expanduser().resolve()


def profile_output_dir(profile: EncodeProfile, src: Optional[Path] = None) -> Path:
    return Path(expand_path_template(profile.output_dir, src)).expanduser().resolve()


def profile_archive_dir(profile: EncodeProfile, src: Optional[Path] = None) -> Path:
    return Path(expand_path_template(profile.archive_dir, src)).expanduser().resolve()


def variant_input_dir(profile: EncodeProfile, variant: OutputVariant) -> Path:
    value = str(getattr(variant, "input_dir", "") or "").strip()
    return Path(value).expanduser().resolve() if value else profile_input_dir(profile)


def variant_output_dir(profile: EncodeProfile, variant: OutputVariant, src: Optional[Path] = None) -> Path:
    value = str(getattr(variant, "output_dir", "") or "").strip()
    return Path(expand_path_template(value, src)).expanduser().resolve() if value else profile_output_dir(profile, src)


def variant_segment_minutes(profile: EncodeProfile, variant: OutputVariant) -> int:
    value = getattr(variant, "segment_minutes", None)
    if value not in (None, ""):
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            pass
    return max(1, int(getattr(profile, "segment_minutes", 10) or 10))


def variant_accepts_source(profile: EncodeProfile, variant: OutputVariant, src: Path) -> bool:
    try:
        return src.parent.resolve() == variant_input_dir(profile, variant)
    except OSError:
        return False


def output_file_stem_for(profile: EncodeProfile, src: Path, variant: OutputVariant) -> str:
    height = "" if variant.height is None else str(variant.height)
    replacements = {
        "${filename}": src.stem,
        "{source}": src.stem,
        "{profile}": variant.name,
        "{profile_id}": variant.id,
        "{folder}": variant.folder_name,
        "{height}": height,
    }
    template = variant.filename_template.strip() or "{source}"
    value = template
    for token, replacement in replacements.items():
        value = value.replace(token, replacement)
    return safe_file_stem(value, default=safe_file_stem(src.stem))


def output_path_for(profile: EncodeProfile, src: Path, variant: OutputVariant) -> Path:
    container = normalize_container_extension(variant.container)
    folder_name = expand_folder_template(variant.folder_name, src)
    return (
        variant_output_dir(profile, variant, src)
        / folder_name
        / f"{output_file_stem_for(profile, src, variant)}.{container}"
    )


def scan_profile_files(profile: EncodeProfile) -> List[FileStatus]:
    if missing_profile_dirs(profile):
        return []

    outputs_by_path: Dict[Path, Dict[str, bool]] = {}
    for variant in profile.outputs:
        if not variant.enabled:
            continue
        input_dir = variant_input_dir(profile, variant)
        if not input_dir.exists():
            continue
        for path in sorted(input_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                continue
            outputs = outputs_by_path.setdefault(path.resolve(), {})
            outputs[variant.id] = output_path_for(profile, path, variant).exists()
    return [
        FileStatus(path=path, outputs=outputs)
        for path, outputs in sorted(outputs_by_path.items(), key=lambda item: str(item[0]).lower())
    ]


def variant_resource_ids(profile: EncodeProfile, variant: OutputVariant) -> List[str]:
    backend = effective_backend(profile, variant)
    resource_by_id = {resource.id: resource for resource in profile.hardware_resources}
    selected = [
        resource_id
        for resource_id in variant.resource_ids
        if resource_id in resource_by_id and resource_backend(resource_id) == backend
    ]
    if selected:
        return selected
    profile_selected = [
        resource_id
        for resource_id in profile.resource_ids
        if resource_id in resource_by_id and resource_backend(resource_id) == backend
    ]
    if profile_selected:
        return profile_selected
    if profile.hardware_resources:
        return []
    fallback = CPU_RESOURCE_ID if backend == BACKEND_CPU else resource_id_for_backend(backend, 0)
    return [fallback]


def build_job_specs(profile: EncodeProfile, files: Iterable[Path]) -> List[JobSpec]:
    specs: List[JobSpec] = []
    next_resource_index: Dict[str, int] = {}
    for src in files:
        src = Path(src)
        for variant in profile.outputs:
            if not variant.enabled:
                continue
            if not variant_accepts_source(profile, variant, src):
                continue
            if not output_path_for(profile, src, variant).exists():
                resources = variant_resource_ids(profile, variant)
                next_index = next_resource_index.get(variant.id, 0)
                assigned = resources[next_index % len(resources)] if resources else ""
                next_resource_index[variant.id] = next_index + 1
                specs.append(
                    JobSpec(
                        src=str(src),
                        profile_id=profile.id,
                        variant_id=variant.id,
                        assigned_resource_id=assigned,
                    )
                )
    return specs


def duplicate_output_targets_for_outputs(outputs: Iterable[OutputVariant]) -> List[str]:
    seen: set[Tuple[str, str, str]] = set()
    duplicates: List[str] = []
    for variant in outputs:
        if not variant.enabled:
            continue
        key = (
            variant.folder_name.lower(),
            normalize_container_extension(variant.container),
            (variant.filename_template or "{source}").strip().lower(),
        )
        if key in seen:
            duplicates.append(f"{variant.folder_name}/{key[2]}.{key[1]}")
        else:
            seen.add(key)
    return duplicates


def duplicate_output_targets(profile: EncodeProfile) -> List[str]:
    return duplicate_output_targets_for_outputs(profile.outputs)


def variant_setting(profile: EncodeProfile, variant: OutputVariant, name: str, default: object = "") -> object:
    value = getattr(variant, name, None)
    if value not in (None, ""):
        return value
    return getattr(profile, name, default)


def variant_use_gpu(profile: EncodeProfile, variant: OutputVariant) -> bool:
    if variant.backend:
        return variant.backend != BACKEND_CPU
    if variant.use_gpu is not None:
        return variant.use_gpu
    return profile.use_gpu


def effective_backend(profile: EncodeProfile, variant: OutputVariant) -> str:
    if variant.backend:
        return normalize_backend(variant.backend)
    return BACKEND_NVENC if variant_use_gpu(profile, variant) else BACKEND_CPU


def variant_gpu_index(profile: EncodeProfile, variant: OutputVariant, resource_id: str = "") -> int:
    if resource_id:
        return resource_index(resource_id)
    resources = variant_resource_ids(profile, variant)
    for item in resources:
        if resource_backend(item) == BACKEND_NVENC:
            return resource_index(item)
    if variant.gpu_index is not None:
        return max(0, variant.gpu_index)
    return max(0, profile.gpu_index)


def variant_audio_codec(profile: EncodeProfile, variant: OutputVariant) -> str:
    value = str(variant_setting(profile, variant, "audio_codec", "copy") or "").strip()
    return normalize_audio_codec(value)


def variant_audio_container(profile: EncodeProfile, variant: OutputVariant) -> str:
    configured = str(variant.audio_container or "").strip()
    return normalize_audio_container_for_codec(variant_audio_codec(profile, variant), configured)


def is_nvenc_codec(codec: str) -> bool:
    return codec.lower().endswith("_nvenc")


def is_qsv_codec(codec: str) -> bool:
    return codec.lower().endswith("_qsv")


def is_amf_codec(codec: str) -> bool:
    return codec.lower().endswith("_amf")


def is_hardware_codec(codec: str) -> bool:
    return is_nvenc_codec(codec) or is_qsv_codec(codec) or is_amf_codec(codec)


def encoder_codec(profile: EncodeProfile, variant: Optional[OutputVariant] = None) -> str:
    if variant is not None:
        if variant.ffmpeg_encoder:
            return variant.ffmpeg_encoder
        backend = effective_backend(profile, variant)
        if backend == BACKEND_NVENC:
            return str(variant_setting(profile, variant, "codec", profile.codec) or "hevc_nvenc")
        if backend == BACKEND_CPU:
            return str(variant_setting(profile, variant, "cpu_codec", profile.cpu_codec) or "libx264")
        return DEFAULT_ENCODER_BY_BACKEND.get(backend, "libx264")
    if profile.use_gpu:
        return profile.codec
    return profile.cpu_codec


def output_pix_fmt(profile: EncodeProfile, variant: Optional[OutputVariant] = None, resource_id: str = "") -> str:
    if is_nvenc_codec(encoder_codec(profile, variant)):
        if variant is not None:
            return str(variant_setting(profile, variant, "pix_fmt", profile.pix_fmt) or "nv12")
        return profile.pix_fmt
    return "yuv420p"


def missing_rate_fields(profile: EncodeProfile, variant: Optional[OutputVariant] = None) -> List[str]:
    if variant is None:
        mode_name = profile.rate_mode.upper()
        bitrate = profile.bitrate
        maxrate = profile.maxrate
        bufsize = profile.bufsize
    else:
        mode_name = str(variant_setting(profile, variant, "rate_mode", profile.rate_mode) or "CQ").upper()
        bitrate = str(variant_setting(profile, variant, "bitrate", profile.bitrate) or "")
        maxrate = str(variant_setting(profile, variant, "maxrate", profile.maxrate) or "")
        bufsize = str(variant_setting(profile, variant, "bufsize", profile.bufsize) or "")
    required: List[Tuple[str, str]] = []
    if mode_name == "VBR":
        required = [("bitrate", bitrate), ("maxrate", maxrate), ("bufsize", bufsize)]
    elif mode_name == "ABR":
        required = [("bitrate", bitrate)]
    elif mode_name == "CBR":
        required = [("bitrate", bitrate), ("bufsize", bufsize)]
    return [name for name, value in required if not str(value or "").strip()]


def validate_rate_settings(profile: EncodeProfile, variant: Optional[OutputVariant] = None) -> None:
    missing = missing_rate_fields(profile, variant)
    if missing:
        mode = (
            profile.rate_mode
            if variant is None
            else str(variant_setting(profile, variant, "rate_mode", profile.rate_mode))
        )
        raise ValueError(f"{mode} requires: {', '.join(missing)}")


def build_video_encoder_args(
    profile: EncodeProfile,
    variant: Optional[OutputVariant] = None,
    resource_id: str = "",
) -> List[str]:
    validate_rate_settings(profile, variant)
    codec = encoder_codec(profile, variant)
    cmd: List[str] = ["-c:v", codec]

    if is_nvenc_codec(codec):
        gpu_index = (
            variant_gpu_index(profile, variant, resource_id) if variant is not None else max(profile.gpu_index, 0)
        )
        preset = str(
            variant_setting(profile, variant, "preset", profile.preset) if variant is not None else profile.preset
        )
        tune = str(variant_setting(profile, variant, "tune", profile.tune) if variant is not None else profile.tune)
        cmd += ["-gpu", str(gpu_index)]
        cmd += ["-preset:v", preset]
        if tune != "none":
            cmd += ["-tune:v", tune]
        if variant is not None:
            split_mode = str(getattr(variant, "split_encode_mode", "") or "").strip()
            if split_mode and split_mode not in {"auto", "default"} and codec.lower() in {"hevc_nvenc", "av1_nvenc"}:
                cmd += ["-split_encode_mode", split_mode]
    elif not is_hardware_codec(codec):
        cpu_preset = str(
            variant_setting(profile, variant, "cpu_preset", profile.cpu_preset)
            if variant is not None
            else profile.cpu_preset
        )
        cpu_tune = str(
            variant_setting(profile, variant, "cpu_tune", profile.cpu_tune) if variant is not None else profile.cpu_tune
        )
        cmd += ["-preset:v", cpu_preset]
        if cpu_tune != "none":
            cmd += ["-tune:v", cpu_tune]

    mode_name = str(
        variant_setting(profile, variant, "rate_mode", profile.rate_mode) if variant is not None else profile.rate_mode
    ).upper()
    cq_value = int(
        variant_setting(profile, variant, "cq_value", profile.cq_value) if variant is not None else profile.cq_value
    )
    bitrate = str(
        variant_setting(profile, variant, "bitrate", profile.bitrate) if variant is not None else profile.bitrate
    )
    maxrate = str(
        variant_setting(profile, variant, "maxrate", profile.maxrate) if variant is not None else profile.maxrate
    )
    bufsize = str(
        variant_setting(profile, variant, "bufsize", profile.bufsize) if variant is not None else profile.bufsize
    )
    if is_nvenc_codec(codec):
        if mode_name == "CQ":
            cmd += ["-b:v", "0", "-cq:v", str(cq_value)]
        elif mode_name == "VBR":
            cmd += [
                "-rc:v",
                "vbr",
                "-b:v",
                bitrate,
                "-maxrate:v",
                maxrate,
                "-bufsize:v",
                bufsize,
                "-cq:v",
                str(cq_value),
            ]
        elif mode_name == "ABR":
            cmd += ["-b:v", bitrate]
        elif mode_name == "CBR":
            cmd += [
                "-rc:v",
                "cbr",
                "-b:v",
                bitrate,
                "-maxrate:v",
                bitrate,
                "-bufsize:v",
                bufsize,
            ]
        else:
            raise ValueError(f"Unknown rate mode: {mode_name}")
    elif is_hardware_codec(codec):
        if mode_name == "CQ":
            raise ValueError(
                f"{codec} does not use the shared CQ/CRF control. Use VBR, ABR, CBR, or backend-specific extra args."
            )
        if mode_name in {"VBR", "ABR"}:
            cmd += ["-b:v", bitrate]
            if mode_name == "VBR":
                cmd += ["-maxrate:v", maxrate, "-bufsize:v", bufsize]
        elif mode_name == "CBR":
            cmd += ["-b:v", bitrate, "-maxrate:v", bitrate, "-bufsize:v", bufsize]
        else:
            raise ValueError(f"Unknown rate mode: {mode_name}")
    else:
        if mode_name == "CQ":
            cmd += ["-crf", str(cq_value)]
        elif mode_name in {"VBR", "ABR"}:
            cmd += ["-b:v", bitrate]
            if mode_name == "VBR":
                cmd += ["-maxrate:v", maxrate, "-bufsize:v", bufsize]
        elif mode_name == "CBR":
            cmd += [
                "-b:v",
                bitrate,
                "-maxrate:v",
                bitrate,
                "-bufsize:v",
                bufsize,
            ]
        else:
            raise ValueError(f"Unknown rate mode: {mode_name}")

    return cmd


def build_ffmpeg_command(
    ffmpeg_path: Path,
    src: Path,
    tmp_out: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
    start_seconds: Optional[float] = None,
    duration_seconds: Optional[float] = None,
    resource_id: str = "",
) -> List[str]:
    cmd: List[str] = [
        str(ffmpeg_path),
        "-hide_banner",
        "-stats_period",
        "1",
    ]
    extra_input_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_input_args", ""))
    extra_video_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_video_args", ""))
    extra_output_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_output_args", ""))

    # Segment resume prioritizes fast input seeking; exact boundaries are handled at segment granularity.
    if start_seconds is not None and start_seconds > 0:
        cmd += ["-ss", format_seconds(start_seconds)]
    if duration_seconds is not None and duration_seconds > 0:
        cmd += ["-t", format_seconds(duration_seconds)]
    cmd += extra_input_args
    cmd += [
        "-i",
        str(src),
        "-y",
        "-map",
        "0:v:0",
        "-an",
        "-pix_fmt",
        output_pix_fmt(profile, variant, resource_id),
    ]
    cmd += build_video_encoder_args(profile, variant, resource_id)

    if variant.height is not None and variant.height > 0:
        scale_flags = str(
            variant_setting(profile, variant, "scale_flags", profile.scale_flags) or "lanczos+accurate_rnd"
        )
        scale = f"scale=-2:{variant.height}:flags={scale_flags}"
        cmd += ["-vf", scale]

    cmd += extra_video_args
    cmd += faststart_args_unless_overridden(variant.container, extra_output_args)
    cmd += extra_output_args
    cmd += [str(tmp_out)]
    return cmd


def build_concat_command(
    ffmpeg_path: Path,
    list_file: Path,
    tmp_out: Path,
    profile: Optional[EncodeProfile] = None,
    variant: Optional[OutputVariant] = None,
) -> List[str]:
    extra_args = parse_ffmpeg_args(
        variant_setting(profile, variant, "extra_concat_args", "") if profile and variant else ""
    )
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-y",
        "-c",
        "copy",
    ]
    cmd += faststart_args_unless_overridden(tmp_out.suffix, extra_args)
    cmd += extra_args
    cmd += [str(tmp_out)]
    return cmd


def build_audio_command(
    ffmpeg_path: Path,
    src: Path,
    tmp_audio: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
) -> List[str]:
    extra_input_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_input_args", ""))
    extra_audio_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_audio_args", ""))
    audio_codec = variant_audio_codec(profile, variant)
    audio_bitrate = str(variant_setting(profile, variant, "audio_bitrate", "") or "").strip()
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-stats_period",
        "1",
    ]
    cmd += extra_input_args
    cmd += [
        "-i",
        str(src),
        "-y",
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        audio_codec,
    ]
    if audio_codec.lower() != "copy" and audio_bitrate:
        cmd += ["-b:a", audio_bitrate]
    cmd += extra_audio_args
    cmd += [str(tmp_audio)]
    return cmd


def build_mux_command(
    ffmpeg_path: Path,
    video_in: Path,
    audio_in: Path,
    tmp_out: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
) -> List[str]:
    extra_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_mux_args", ""))
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-i",
        str(video_in),
        "-i",
        str(audio_in),
        "-y",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-shortest",
    ]
    cmd += faststart_args_unless_overridden(tmp_out.suffix, extra_args)
    cmd += extra_args
    cmd += [str(tmp_out)]
    return cmd


def command_to_text(cmd: List[str]) -> str:
    def quote(x: str) -> str:
        if any(ch in x for ch in " ()&[]{}^=;!'+,`~"):
            return '"' + x.replace('"', '\\"') + '"'
        return x

    return " ".join(quote(str(x)) for x in cmd)


def format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms >= 1000:
        whole += ms // 1000
        ms %= 1000
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def parse_ffmpeg_time(line: str) -> Optional[float]:
    match = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", line)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def probe_duration(ffprobe_path: Path, src: Path) -> Optional[float]:
    if not ffprobe_path.exists():
        return None
    try:
        result = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(src),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    if duration <= 0:
        return None
    return duration


def probe_has_audio(ffprobe_path: Path, src: Path) -> bool:
    if not ffprobe_path.exists():
        return False
    try:
        result = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(src),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(f"ffprobe audio probe failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"ffprobe audio probe failed: exit {result.returncode}{suffix}")
    return bool(result.stdout.strip())


def segment_ranges(duration: Optional[float], segment_seconds: int) -> List[Tuple[float, Optional[float]]]:
    if duration is None or segment_seconds <= 0:
        return [(0.0, None)]

    segment_seconds = max(segment_seconds, 30)
    count = max(1, math.ceil(duration / segment_seconds))
    ranges: List[Tuple[float, Optional[float]]] = []
    for index in range(count):
        start = index * segment_seconds
        remaining = max(duration - start, 0.0)
        ranges.append((float(start), min(float(segment_seconds), remaining)))
    return ranges


def job_fingerprint(profile: EncodeProfile, variant: OutputVariant) -> str:
    data = {
        "profile_id": profile.id,
        "variant_id": variant.id,
        "variant_name": variant.name,
        "variant_folder": variant.folder_name,
        "variant_height": variant.height,
        "variant_container": variant.container,
        "variant_filename_template": variant.filename_template,
        "variant_input_dir": str(variant_input_dir(profile, variant)),
        "variant_output_dir": str(variant_output_dir(profile, variant)),
        "segment_minutes": variant_segment_minutes(profile, variant),
        "backend": variant.backend,
        "ffmpeg_encoder": encoder_codec(profile, variant),
        "resource_ids": variant_resource_ids(profile, variant),
        "split_encode_mode": variant.split_encode_mode,
        "use_gpu": variant_use_gpu(profile, variant),
        "gpu_index": variant_gpu_index(profile, variant),
        "codec": encoder_codec(profile, variant),
        "cpu_codec": variant_setting(profile, variant, "cpu_codec", profile.cpu_codec),
        "preset": variant_setting(profile, variant, "preset", profile.preset),
        "cpu_preset": variant_setting(profile, variant, "cpu_preset", profile.cpu_preset),
        "cpu_tune": variant_setting(profile, variant, "cpu_tune", profile.cpu_tune),
        "tune": variant_setting(profile, variant, "tune", profile.tune),
        "rate_mode": variant_setting(profile, variant, "rate_mode", profile.rate_mode),
        "cq_value": variant_setting(profile, variant, "cq_value", profile.cq_value),
        "bitrate": variant_setting(profile, variant, "bitrate", profile.bitrate),
        "maxrate": variant_setting(profile, variant, "maxrate", profile.maxrate),
        "bufsize": variant_setting(profile, variant, "bufsize", profile.bufsize),
        "pix_fmt": output_pix_fmt(profile, variant),
        "scale_flags": variant_setting(profile, variant, "scale_flags", profile.scale_flags),
        "audio_codec": variant_audio_codec(profile, variant),
        "audio_bitrate": variant_setting(profile, variant, "audio_bitrate", ""),
        "audio_container": variant_audio_container(profile, variant),
        "extra_input_args": variant_setting(profile, variant, "extra_input_args", ""),
        "extra_video_args": variant_setting(profile, variant, "extra_video_args", ""),
        "extra_audio_args": variant_setting(profile, variant, "extra_audio_args", ""),
        "extra_output_args": variant_setting(profile, variant, "extra_output_args", ""),
        "extra_concat_args": variant_setting(profile, variant, "extra_concat_args", ""),
        "extra_mux_args": variant_setting(profile, variant, "extra_mux_args", ""),
    }
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:10]


def source_fingerprint(src: Path) -> str:
    data: Dict[str, Any] = {"path": str(src.resolve())}
    try:
        stat = src.stat()
    except OSError:
        pass
    else:
        data["size"] = stat.st_size
        data["mtime_ns"] = stat.st_mtime_ns
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8", errors="ignore")).hexdigest()[:10]


def short_path_label(value: str, limit: int) -> str:
    label = safe_folder_name(value)
    if len(label) <= limit:
        return label
    return label[:limit].rstrip(".- ") or "item"


def job_key(src: Path, profile: EncodeProfile, variant: OutputVariant) -> str:
    src_digest = source_fingerprint(src)
    settings_digest = job_fingerprint(profile, variant)
    stem = short_path_label(src.stem, 80)
    variant_id = short_path_label(variant.id, 40)
    return f"{stem}-{variant_id}-{src_digest}-{settings_digest}"


def segment_dir_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant) -> Path:
    return paths.tmp_dir / "segments" / job_key(src, profile, variant)


def segment_file_name(
    variant: OutputVariant,
    index: int,
    partial: bool = False,
    src: Optional[Path] = None,
) -> str:
    container = normalize_container_extension(variant.container)
    marker = ".partial" if partial else ""
    if src is not None:
        stem = safe_file_stem(src.stem)
        if len(stem) > SEGMENT_SOURCE_STEM_LIMIT:
            stem = stem[:SEGMENT_SOURCE_STEM_LIMIT].rstrip(".- ") or "video"
        return f"{stem}-{index + 1:03d}{marker}.{container}"
    return f"segment-{index:05d}{marker}.{container}"


def joined_video_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant) -> Path:
    container = normalize_container_extension(variant.container)
    return paths.tmp_dir / f"{job_key(src, profile, variant)}.video.{container}"


def temp_audio_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant) -> Path:
    container = variant_audio_container(profile, variant)
    return paths.tmp_dir / f"{job_key(src, profile, variant)}.audio.{container}"


def temp_output_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant) -> Path:
    container = normalize_container_extension(variant.container)
    return paths.tmp_dir / f"{job_key(src, profile, variant)}.final.{container}"


def write_concat_file(list_path: Path, segments: List[Path]) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for segment in segments:
        value = escape_concat_path(segment)
        lines.append(f"file '{value}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_concat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("\\", "\\\\").replace("'", "\\'")


def save_state(paths: AppPaths, profile: EncodeProfile, specs: List[JobSpec]) -> None:
    ensure_dirs(paths)
    data = {
        "version": 3,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile_to_dict(profile),
        "jobs": [asdict(spec) for spec in specs],
    }
    tmp_file = paths.state_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(paths.state_file)


def load_state(paths: AppPaths) -> Optional[Dict[str, Any]]:
    if not paths.state_file.exists():
        return None
    return read_json_object(paths.state_file)


def clear_state(paths: AppPaths) -> None:
    if paths.state_file.exists():
        paths.state_file.unlink()


def profile_from_state(
    data: Dict[str, Any],
    paths: Optional[AppPaths] = None,
    gpus: Optional[List[GpuInfo]] = None,
) -> EncodeProfile:
    return EncodeProfile.from_dict(data.get("profile", {}), paths, gpus)


def resumable_specs(profile: EncodeProfile, data: Dict[str, Any]) -> List[JobSpec]:
    specs: List[JobSpec] = []
    for item in data.get("jobs", []):
        if not isinstance(item, dict):
            continue
        try:
            spec = JobSpec(
                src=str(item["src"]),
                profile_id=str(item["profile_id"]),
                variant_id=str(item["variant_id"]),
                assigned_resource_id=str(item.get("assigned_resource_id", "")),
                assigned_slot=normalize_int(item.get("assigned_slot", 0), minimum=0, default=0),
            )
        except KeyError:
            continue
        src = Path(spec.src)
        if not src.exists():
            continue
        try:
            variant = variant_by_id(profile, spec.variant_id)
        except ValueError:
            continue
        if not output_path_for(profile, src, variant).exists():
            specs.append(spec)
    return specs
