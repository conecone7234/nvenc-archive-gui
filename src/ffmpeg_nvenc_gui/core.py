from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

VIDEO_EXTS = {".mkv", ".m2ts", ".mp4"}


@dataclass
class AppPaths:
    base_dir: Path
    ffmpeg_path: Path
    obs_dir: Path
    source_dir: Path
    mp4_dir: Path
    k4_dir: Path
    tmp_dir: Path
    log_dir: Path
    state_file: Path
    download_dir: Path


@dataclass
class EncodeSettings:
    max_jobs: int = 2
    make_4k: bool = True
    make_mp4: bool = True
    height_4k: int = 2160
    codec: str = "hevc_nvenc"
    preset: str = "p7"
    tune: str = "hq"
    rate_mode: str = "CQ"
    cq_value: int = 15
    bitrate: str = "35000k"
    maxrate: str = "50000k"
    bufsize: str = "100000k"
    pix_fmt: str = "nv12"
    scale_flags: str = "lanczos+accurate_rnd"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EncodeSettings":
        base = asdict(EncodeSettings())
        base.update(data or {})
        return EncodeSettings(**base)


@dataclass
class JobSpec:
    src: str
    mode: str


@dataclass
class FileStatus:
    path: Path
    has_4k: bool
    has_mp4: bool

    @property
    def label(self) -> str:
        if self.has_4k and self.has_mp4:
            return "[done]"
        if self.has_4k and not self.has_mp4:
            return "[missing MP4]"
        if not self.has_4k and self.has_mp4:
            return "[missing 4K]"
        return "[new]"


def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def build_paths(base_dir: Optional[Path] = None) -> AppPaths:
    base = (base_dir or get_base_dir()).resolve()
    tmp = base / "tmp"
    log_dir = tmp / "logs"
    return AppPaths(
        base_dir=base,
        ffmpeg_path=base / "ffmpeg" / "ffmpeg.exe",
        obs_dir=base / "OBSData",
        source_dir=base / "SourceData",
        mp4_dir=base / "SourceData(MP4)",
        k4_dir=base / "SourceData(4K)",
        tmp_dir=tmp,
        log_dir=log_dir,
        state_file=tmp / "encoder_state.json",
        download_dir=tmp / "downloads",
    )


def ensure_dirs(paths: AppPaths) -> None:
    paths.obs_dir.mkdir(parents=True, exist_ok=True)
    paths.source_dir.mkdir(parents=True, exist_ok=True)
    paths.mp4_dir.mkdir(parents=True, exist_ok=True)
    paths.k4_dir.mkdir(parents=True, exist_ok=True)
    paths.tmp_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.download_dir.mkdir(parents=True, exist_ok=True)
    paths.ffmpeg_path.parent.mkdir(parents=True, exist_ok=True)


def scan_obs_files(paths: AppPaths) -> List[FileStatus]:
    if not paths.obs_dir.exists():
        return []

    items: List[FileStatus] = []
    for path in sorted(paths.obs_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        stem = path.stem
        items.append(
            FileStatus(
                path=path,
                has_4k=(paths.k4_dir / f"{stem}.mp4").exists(),
                has_mp4=(paths.mp4_dir / f"{stem}.mp4").exists(),
            )
        )
    return items


def output_path_for(paths: AppPaths, src: Path, mode: str) -> Path:
    if mode.upper() == "4K":
        return paths.k4_dir / f"{src.stem}.mp4"
    if mode.upper() == "MP4":
        return paths.mp4_dir / f"{src.stem}.mp4"
    raise ValueError(f"Unknown mode: {mode}")


def temp_path_for(paths: AppPaths, src: Path, mode: str) -> Path:
    safe_mode = mode.replace(" ", "_")
    return paths.tmp_dir / f"{src.stem}-{safe_mode}.mp4"


def build_job_specs(paths: AppPaths, files: Iterable[Path], settings: EncodeSettings) -> List[JobSpec]:
    specs: List[JobSpec] = []

    for src in files:
        src = Path(src)
        if settings.make_4k and not output_path_for(paths, src, "4K").exists():
            specs.append(JobSpec(src=str(src), mode="4K"))
        if settings.make_mp4 and not output_path_for(paths, src, "MP4").exists():
            specs.append(JobSpec(src=str(src), mode="MP4"))

    return specs


def build_ffmpeg_command(
    ffmpeg_path: Path,
    src: Path,
    tmp_out: Path,
    mode: str,
    settings: EncodeSettings,
) -> List[str]:
    cmd: List[str] = [
        str(ffmpeg_path),
        "-hide_banner",
        "-stats_period",
        "1",
        "-i",
        str(src),
        "-y",
        "-pix_fmt",
        settings.pix_fmt,
        "-preset:v",
        settings.preset,
    ]

    if settings.tune != "none":
        cmd += ["-tune:v", settings.tune]

    mode_name = settings.rate_mode.upper()
    if mode_name == "CQ":
        cmd += ["-b:v", "0", "-cq:v", str(settings.cq_value)]
    elif mode_name == "VBR":
        cmd += [
            "-rc:v",
            "vbr",
            "-b:v",
            settings.bitrate,
            "-maxrate:v",
            settings.maxrate,
            "-bufsize:v",
            settings.bufsize,
            "-cq:v",
            str(settings.cq_value),
        ]
    elif mode_name == "ABR":
        cmd += ["-b:v", settings.bitrate]
    elif mode_name == "CBR":
        cmd += [
            "-rc:v",
            "cbr",
            "-b:v",
            settings.bitrate,
            "-maxrate:v",
            settings.bitrate,
            "-bufsize:v",
            settings.bufsize,
        ]
    else:
        raise ValueError(f"Unknown rate mode: {settings.rate_mode}")

    cmd += ["-c:v", settings.codec, "-movflags", "+faststart"]

    if mode.upper() == "4K":
        scale = f"scale=-1:{settings.height_4k}:flags={settings.scale_flags}"
        cmd += ["-vf", scale]
    elif mode.upper() != "MP4":
        raise ValueError(f"Unknown mode: {mode}")

    cmd += ["-c:a", "copy", "-c:s", "copy", "-map", "0", str(tmp_out)]
    return cmd


def command_to_text(cmd: List[str]) -> str:
    def quote(x: str) -> str:
        if any(ch in x for ch in " ()&[]{}^=;!'+,`~"):
            return '"' + x.replace('"', '\\"') + '"'
        return x

    return " ".join(quote(str(x)) for x in cmd)


def save_state(paths: AppPaths, settings: EncodeSettings, specs: List[JobSpec]) -> None:
    ensure_dirs(paths)
    data = {
        "version": 1,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": asdict(settings),
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


def resumable_specs(paths: AppPaths, data: Dict[str, Any]) -> List[JobSpec]:
    specs: List[JobSpec] = []
    for item in data.get("jobs", []):
        src = Path(item["src"])
        mode = item["mode"]
        out_file = output_path_for(paths, src, mode)
        if not out_file.exists():
            specs.append(JobSpec(src=str(src), mode=mode))
    return specs
