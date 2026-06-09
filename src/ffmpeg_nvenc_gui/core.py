from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
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
        if not self.id:
            self.id = new_id("variant")
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
        base.update(data or {})
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

    @staticmethod
    def from_dict(
        data: Dict[str, Any],
        paths: Optional[AppPaths] = None,
        gpus: Optional[List[GpuInfo]] = None,
    ) -> "EncodeProfile":
        base_paths = paths or build_paths()
        default_gpus = [] if gpus is None else gpus
        base = asdict(default_profile(base_paths, default_gpus))
        base.update(data or {})
        base["outputs"] = [OutputVariant.from_dict(x) for x in base.get("outputs", [])]
        if not base["outputs"]:
            base["outputs"] = default_outputs()
        return EncodeProfile(**base)


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


def ensure_profile_dirs(profile: EncodeProfile) -> None:
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


def load_profiles(paths: AppPaths, gpus: Optional[List[GpuInfo]] = None) -> List[EncodeProfile]:
    if not paths.config_file.exists():
        return [default_profile(paths, gpus)]

    data = json.loads(paths.config_file.read_text(encoding="utf-8"))
    profiles = [EncodeProfile.from_dict(item, paths, gpus) for item in data.get("profiles", [])]
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


def build_video_encoder_args(profile: EncodeProfile) -> List[str]:
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


def job_key(src: Path, variant: OutputVariant) -> str:
    digest = hashlib.sha1(str(src.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return safe_folder_name(f"{src.stem}-{variant.id}-{digest}")


def segment_dir_for(paths: AppPaths, src: Path, variant: OutputVariant) -> Path:
    return paths.tmp_dir / "segments" / job_key(src, variant)


def segment_path_for(paths: AppPaths, src: Path, variant: OutputVariant, index: int) -> Path:
    container = normalize_container_extension(variant.container)
    return segment_dir_for(paths, src, variant) / f"segment-{index:05d}.{container}"


def partial_segment_path_for(paths: AppPaths, src: Path, variant: OutputVariant, index: int) -> Path:
    container = normalize_container_extension(variant.container)
    return segment_dir_for(paths, src, variant) / f"segment-{index:05d}.partial.{container}"


def temp_output_path_for(paths: AppPaths, src: Path, variant: OutputVariant) -> Path:
    container = normalize_container_extension(variant.container)
    return paths.tmp_dir / f"{job_key(src, variant)}.final.{container}"


def concat_list_path_for(paths: AppPaths, src: Path, variant: OutputVariant) -> Path:
    return segment_dir_for(paths, src, variant) / "concat.txt"


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
    return json.loads(paths.state_file.read_text(encoding="utf-8"))


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
