from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
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


@dataclass
class OutputVariant:
    id: str
    name: str
    folder_name: str
    height: Optional[int] = None
    container: str = "mp4"
    enabled: bool = True
    filename_template: str = "{source}"
    use_gpu: Optional[bool] = None
    gpu_index: Optional[int] = None
    gpu_name: str = ""
    codec: str = ""
    cpu_codec: str = ""
    preset: str = ""
    cpu_preset: str = ""
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
        self.use_gpu = normalize_optional_bool(self.use_gpu)
        self.gpu_index = normalize_optional_int(self.gpu_index, minimum=0)
        self.gpu_name = str(self.gpu_name or "").strip()
        self.codec = str(self.codec or "").strip()
        self.cpu_codec = str(self.cpu_codec or "").strip()
        self.preset = str(self.preset or "").strip()
        self.cpu_preset = str(self.cpu_preset or "").strip()
        self.tune = str(self.tune or "").strip()
        self.rate_mode = str(self.rate_mode or "").strip().upper()
        self.cq_value = normalize_optional_int(self.cq_value)
        self.bitrate = str(self.bitrate or "").strip()
        self.maxrate = str(self.maxrate or "").strip()
        self.bufsize = str(self.bufsize or "").strip()
        self.pix_fmt = str(self.pix_fmt or "").strip()
        self.scale_flags = str(self.scale_flags or "").strip()
        self.audio_codec = str(self.audio_codec or "").strip()
        self.audio_bitrate = str(self.audio_bitrate or "").strip()
        self.audio_container = normalize_container_extension(self.audio_container, default="") or ""
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
        }
        base.update(dataclass_values(OutputVariant, data))
        return OutputVariant(**base)


@dataclass
class EncodeProfile:
    id: str
    name: str
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
    rate_mode: str = "CQ"
    cq_value: int = 18
    bitrate: str = "25000k"
    maxrate: str = "40000k"
    bufsize: str = "80000k"
    pix_fmt: str = "nv12"
    scale_flags: str = "lanczos+accurate_rnd"
    outputs: List[OutputVariant] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip() or new_id("profile")
        self.name = str(self.name or "").strip() or self.id
        self.input_dir = str(self.input_dir or "").strip()
        self.output_dir = str(self.output_dir or "").strip()
        self.archive_dir = str(self.archive_dir or "").strip()
        self.gpu_name = str(self.gpu_name or "").strip()
        self.codec = str(self.codec or "hevc_nvenc").strip()
        self.cpu_codec = str(self.cpu_codec or "libx264").strip()
        self.preset = str(self.preset or "p7").strip()
        self.cpu_preset = str(self.cpu_preset or "medium").strip()
        self.tune = str(self.tune or "hq").strip()
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
        base.update(dataclass_values(EncodeProfile, data))
        raw_outputs = base.get("outputs")
        if not isinstance(raw_outputs, list):
            raw_outputs = []
        base["outputs"] = [OutputVariant.from_dict(x) for x in raw_outputs]
        if not base["outputs"]:
            base["outputs"] = default_outputs()
        return normalize_profile_gpu(EncodeProfile(**base), default_gpus)


@dataclass
class JobSpec:
    src: str
    profile_id: str
    variant_id: str


@dataclass
class FileStatus:
    path: Path
    outputs: Dict[str, bool]

    def label(self, profile: EncodeProfile) -> str:
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
    output_dir = profile_output_dir(profile)
    archive_dir = profile_archive_dir(profile)

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    for variant in profile.outputs:
        if variant.enabled:
            (output_dir / variant.folder_name).mkdir(parents=True, exist_ok=True)


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
            name="Master 2160p",
            folder_name="master-2160p",
            height=2160,
            container="mp4",
        ),
        OutputVariant(
            id="reference_1080p",
            name="Reference 1080p",
            folder_name="reference-1080p",
            height=1080,
            container="mp4",
        ),
    ]


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


def normalize_profile_gpu(profile: EncodeProfile, gpus: Optional[List[GpuInfo]]) -> EncodeProfile:
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
        if variant.use_gpu is not True:
            continue
        gpu = gpu_by_index.get(variant.gpu_index if variant.gpu_index is not None else profile.gpu_index)
        if gpu is None:
            variant.use_gpu = False
            variant.gpu_index = 0
            variant.gpu_name = ""
        else:
            variant.gpu_index = gpu.index
            variant.gpu_name = gpu.name
    return profile


def default_profile(paths: AppPaths, gpus: Optional[List[GpuInfo]] = None) -> EncodeProfile:
    detected = detect_nvidia_gpus() if gpus is None else gpus
    gpu = detected[0] if detected else None
    return EncodeProfile(
        id="default",
        name="Archive Profile",
        input_dir=str(paths.base_dir / "Incoming"),
        output_dir=str(paths.base_dir / "Encoded"),
        archive_dir=str(paths.base_dir / "SourceArchive"),
        max_parallel_jobs=2 if gpu else 1,
        segment_minutes=10,
        use_gpu=gpu is not None,
        gpu_index=gpu.index if gpu else 0,
        gpu_name=gpu.name if gpu else "",
        codec="hevc_nvenc",
        cpu_codec="libx264",
        outputs=default_outputs(),
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
        "version": 1,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profiles": [asdict(profile) for profile in profiles],
    }
    tmp_file = paths.config_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(paths.config_file)


def variant_by_id(profile: EncodeProfile, variant_id: str) -> OutputVariant:
    for variant in profile.outputs:
        if variant.id == variant_id:
            return variant
    raise ValueError(f"Unknown output variant: {variant_id}")


def profile_input_dir(profile: EncodeProfile) -> Path:
    return Path(profile.input_dir).expanduser().resolve()


def profile_output_dir(profile: EncodeProfile) -> Path:
    return Path(profile.output_dir).expanduser().resolve()


def profile_archive_dir(profile: EncodeProfile) -> Path:
    return Path(profile.archive_dir).expanduser().resolve()


def output_file_stem_for(profile: EncodeProfile, src: Path, variant: OutputVariant) -> str:
    height = "" if variant.height is None else str(variant.height)
    replacements = {
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
    return profile_output_dir(profile) / variant.folder_name / f"{output_file_stem_for(profile, src, variant)}.{container}"


def scan_profile_files(profile: EncodeProfile) -> List[FileStatus]:
    if missing_profile_dirs(profile):
        return []
    input_dir = profile_input_dir(profile)
    if not input_dir.exists():
        return []

    items: List[FileStatus] = []
    for path in sorted(input_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        outputs = {
            variant.id: output_path_for(profile, path, variant).exists()
            for variant in profile.outputs
            if variant.enabled
        }
        items.append(FileStatus(path=path, outputs=outputs))
    return items


def build_job_specs(profile: EncodeProfile, files: Iterable[Path]) -> List[JobSpec]:
    specs: List[JobSpec] = []
    for src in files:
        src = Path(src)
        for variant in profile.outputs:
            if not variant.enabled:
                continue
            if not output_path_for(profile, src, variant).exists():
                specs.append(JobSpec(src=str(src), profile_id=profile.id, variant_id=variant.id))
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
    if variant.use_gpu is not None:
        return variant.use_gpu
    return profile.use_gpu


def variant_gpu_index(profile: EncodeProfile, variant: OutputVariant) -> int:
    if variant.gpu_index is not None:
        return max(0, variant.gpu_index)
    return max(0, profile.gpu_index)


def variant_audio_codec(profile: EncodeProfile, variant: OutputVariant) -> str:
    value = str(variant_setting(profile, variant, "audio_codec", "copy") or "").strip()
    return value or "copy"


def variant_audio_container(profile: EncodeProfile, variant: OutputVariant) -> str:
    configured = str(variant.audio_container or "").strip()
    if configured:
        return normalize_container_extension(configured, default="mka")
    codec = variant_audio_codec(profile, variant).lower()
    if codec in {"aac", "alac"}:
        return "m4a"
    if codec in {"mp3", "libmp3lame"}:
        return "mp3"
    return "mka"


def is_nvenc_codec(codec: str) -> bool:
    return codec.lower().endswith("_nvenc")


def encoder_codec(profile: EncodeProfile, variant: Optional[OutputVariant] = None) -> str:
    if variant is not None:
        if variant_use_gpu(profile, variant):
            return str(variant_setting(profile, variant, "codec", profile.codec) or "hevc_nvenc")
        return str(variant_setting(profile, variant, "cpu_codec", profile.cpu_codec) or "libx264")
    if profile.use_gpu:
        return profile.codec
    return profile.cpu_codec


def output_pix_fmt(profile: EncodeProfile, variant: Optional[OutputVariant] = None) -> str:
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
        mode = profile.rate_mode if variant is None else str(variant_setting(profile, variant, "rate_mode", profile.rate_mode))
        raise ValueError(f"{mode} requires: {', '.join(missing)}")


def build_video_encoder_args(profile: EncodeProfile, variant: Optional[OutputVariant] = None) -> List[str]:
    validate_rate_settings(profile, variant)
    codec = encoder_codec(profile, variant)
    cmd: List[str] = ["-c:v", codec]

    if is_nvenc_codec(codec):
        gpu_index = variant_gpu_index(profile, variant) if variant is not None else max(profile.gpu_index, 0)
        preset = str(variant_setting(profile, variant, "preset", profile.preset) if variant is not None else profile.preset)
        tune = str(variant_setting(profile, variant, "tune", profile.tune) if variant is not None else profile.tune)
        cmd += ["-gpu", str(gpu_index)]
        cmd += ["-preset:v", preset]
        if tune != "none":
            cmd += ["-tune:v", tune]
    else:
        cpu_preset = str(
            variant_setting(profile, variant, "cpu_preset", profile.cpu_preset) if variant is not None else profile.cpu_preset
        )
        cmd += ["-preset:v", cpu_preset]

    mode_name = str(variant_setting(profile, variant, "rate_mode", profile.rate_mode) if variant is not None else profile.rate_mode).upper()
    cq_value = int(variant_setting(profile, variant, "cq_value", profile.cq_value) if variant is not None else profile.cq_value)
    bitrate = str(variant_setting(profile, variant, "bitrate", profile.bitrate) if variant is not None else profile.bitrate)
    maxrate = str(variant_setting(profile, variant, "maxrate", profile.maxrate) if variant is not None else profile.maxrate)
    bufsize = str(variant_setting(profile, variant, "bufsize", profile.bufsize) if variant is not None else profile.bufsize)
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
        output_pix_fmt(profile, variant),
    ]
    cmd += build_video_encoder_args(profile, variant)

    if variant.height is not None and variant.height > 0:
        scale_flags = str(variant_setting(profile, variant, "scale_flags", profile.scale_flags) or "lanczos+accurate_rnd")
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
    extra_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_concat_args", "") if profile and variant else "")
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
        return True
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
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except Exception:
        return True
    return result.returncode == 0 and bool(result.stdout.strip())


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
        "segment_minutes": profile.segment_minutes,
        "use_gpu": variant_use_gpu(profile, variant),
        "gpu_index": variant_gpu_index(profile, variant),
        "codec": encoder_codec(profile, variant),
        "cpu_codec": variant_setting(profile, variant, "cpu_codec", profile.cpu_codec),
        "preset": variant_setting(profile, variant, "preset", profile.preset),
        "cpu_preset": variant_setting(profile, variant, "cpu_preset", profile.cpu_preset),
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
        "version": 2,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": asdict(profile),
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
