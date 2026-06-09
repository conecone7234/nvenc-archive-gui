from __future__ import annotations

import hashlib
import json
import math
import re
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


def normalize_container_extension(value: object, default: str = "mp4") -> str:
    candidate = str(value or "").strip().lower().lstrip(".")
    if SAFE_CONTAINER_RE.fullmatch(candidate):
        return candidate
    return default


def faststart_args_for_container(value: object) -> List[str]:
    container = normalize_container_extension(value)
    if container in FASTSTART_CONTAINERS:
        return ["-movflags", "+faststart"]
    return []


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
    if not profile.use_gpu:
        return profile

    for gpu in gpus or []:
        if gpu.index == profile.gpu_index:
            profile.gpu_name = gpu.name
            return profile

    profile.use_gpu = False
    profile.gpu_index = 0
    profile.gpu_name = ""
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


def output_path_for(profile: EncodeProfile, src: Path, variant: OutputVariant) -> Path:
    container = normalize_container_extension(variant.container)
    return profile_output_dir(profile) / variant.folder_name / f"{src.stem}.{container}"


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
    seen: set[Tuple[str, str]] = set()
    duplicates: List[str] = []
    for variant in outputs:
        if not variant.enabled:
            continue
        key = (variant.folder_name.lower(), normalize_container_extension(variant.container))
        if key in seen:
            duplicates.append(f"{variant.folder_name}.{key[1]}")
        else:
            seen.add(key)
    return duplicates


def duplicate_output_targets(profile: EncodeProfile) -> List[str]:
    return duplicate_output_targets_for_outputs(profile.outputs)


def is_nvenc_codec(codec: str) -> bool:
    return codec.lower().endswith("_nvenc")


def encoder_codec(profile: EncodeProfile) -> str:
    if profile.use_gpu:
        return profile.codec
    return profile.cpu_codec


def output_pix_fmt(profile: EncodeProfile) -> str:
    if is_nvenc_codec(encoder_codec(profile)):
        return profile.pix_fmt
    return "yuv420p"


def missing_rate_fields(profile: EncodeProfile) -> List[str]:
    mode_name = profile.rate_mode.upper()
    required: List[Tuple[str, str]] = []
    if mode_name == "VBR":
        required = [("bitrate", profile.bitrate), ("maxrate", profile.maxrate), ("bufsize", profile.bufsize)]
    elif mode_name == "ABR":
        required = [("bitrate", profile.bitrate)]
    elif mode_name == "CBR":
        required = [("bitrate", profile.bitrate), ("bufsize", profile.bufsize)]
    return [name for name, value in required if not str(value or "").strip()]


def validate_rate_settings(profile: EncodeProfile) -> None:
    missing = missing_rate_fields(profile)
    if missing:
        raise ValueError(f"{profile.rate_mode} requires: {', '.join(missing)}")


def build_video_encoder_args(profile: EncodeProfile) -> List[str]:
    validate_rate_settings(profile)
    codec = encoder_codec(profile)
    cmd: List[str] = ["-c:v", codec]

    if is_nvenc_codec(codec):
        cmd += ["-gpu", str(max(profile.gpu_index, 0))]
        cmd += ["-preset:v", profile.preset]
        if profile.tune != "none":
            cmd += ["-tune:v", profile.tune]
    else:
        cmd += ["-preset:v", profile.cpu_preset]

    mode_name = profile.rate_mode.upper()
    if is_nvenc_codec(codec):
        if mode_name == "CQ":
            cmd += ["-b:v", "0", "-cq:v", str(profile.cq_value)]
        elif mode_name == "VBR":
            cmd += [
                "-rc:v",
                "vbr",
                "-b:v",
                profile.bitrate,
                "-maxrate:v",
                profile.maxrate,
                "-bufsize:v",
                profile.bufsize,
                "-cq:v",
                str(profile.cq_value),
            ]
        elif mode_name == "ABR":
            cmd += ["-b:v", profile.bitrate]
        elif mode_name == "CBR":
            cmd += [
                "-rc:v",
                "cbr",
                "-b:v",
                profile.bitrate,
                "-maxrate:v",
                profile.bitrate,
                "-bufsize:v",
                profile.bufsize,
            ]
        else:
            raise ValueError(f"Unknown rate mode: {profile.rate_mode}")
    else:
        if mode_name == "CQ":
            cmd += ["-crf", str(profile.cq_value)]
        elif mode_name in {"VBR", "ABR"}:
            cmd += ["-b:v", profile.bitrate]
            if mode_name == "VBR":
                cmd += ["-maxrate:v", profile.maxrate, "-bufsize:v", profile.bufsize]
        elif mode_name == "CBR":
            cmd += [
                "-b:v",
                profile.bitrate,
                "-maxrate:v",
                profile.bitrate,
                "-bufsize:v",
                profile.bufsize,
            ]
        else:
            raise ValueError(f"Unknown rate mode: {profile.rate_mode}")

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

    # Segment resume prioritizes fast input seeking; exact boundaries are handled at segment granularity.
    if start_seconds is not None and start_seconds > 0:
        cmd += ["-ss", format_seconds(start_seconds)]
    if duration_seconds is not None and duration_seconds > 0:
        cmd += ["-t", format_seconds(duration_seconds)]
    cmd += [
        "-i",
        str(src),
        "-y",
        "-map",
        "0",
        "-pix_fmt",
        output_pix_fmt(profile),
    ]
    cmd += build_video_encoder_args(profile)

    if variant.height is not None and variant.height > 0:
        scale = f"scale=-2:{variant.height}:flags={profile.scale_flags}"
        cmd += ["-vf", scale]

    cmd += ["-c:a", "copy", "-c:s", "copy"]
    cmd += faststart_args_for_container(variant.container)
    cmd += [str(tmp_out)]
    return cmd


def build_concat_command(ffmpeg_path: Path, list_file: Path, tmp_out: Path) -> List[str]:
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
    cmd += faststart_args_for_container(tmp_out.suffix)
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
        "segment_minutes": profile.segment_minutes,
        "use_gpu": profile.use_gpu,
        "gpu_index": profile.gpu_index,
        "codec": profile.codec,
        "cpu_codec": profile.cpu_codec,
        "preset": profile.preset,
        "cpu_preset": profile.cpu_preset,
        "tune": profile.tune,
        "rate_mode": profile.rate_mode,
        "cq_value": profile.cq_value,
        "bitrate": profile.bitrate,
        "maxrate": profile.maxrate,
        "bufsize": profile.bufsize,
        "pix_fmt": profile.pix_fmt,
        "scale_flags": profile.scale_flags,
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


def segment_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant, index: int) -> Path:
    container = normalize_container_extension(variant.container)
    return segment_dir_for(paths, src, profile, variant) / f"segment-{index:05d}.{container}"


def partial_segment_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant, index: int) -> Path:
    container = normalize_container_extension(variant.container)
    return segment_dir_for(paths, src, profile, variant) / f"segment-{index:05d}.partial.{container}"


def temp_output_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant) -> Path:
    container = normalize_container_extension(variant.container)
    return paths.tmp_dir / f"{job_key(src, profile, variant)}.final.{container}"


def concat_list_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant) -> Path:
    return segment_dir_for(paths, src, profile, variant) / "concat.txt"


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
