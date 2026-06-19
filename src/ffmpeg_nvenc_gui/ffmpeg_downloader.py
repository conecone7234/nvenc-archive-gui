from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Optional

from .core import AppPaths, ensure_dirs, is_nvenc_codec, resource_index

ProgressCallback = Optional[Callable[[str], None]]

GYAN_FFMPEG_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
COMMON_ENCODERS = {
    "libx264",
    "libx265",
    "h264_nvenc",
    "hevc_nvenc",
    "av1_nvenc",
    "h264_qsv",
    "hevc_qsv",
    "av1_qsv",
    "h264_amf",
    "hevc_amf",
    "av1_amf",
}


class FfmpegDownloadError(RuntimeError):
    pass


def find_binary_member(names: Iterable[str], binary_name: str) -> Optional[str]:
    candidates = []
    for name in names:
        normalized = name.replace("\\", "/")
        if normalized.lower().endswith(f"/bin/{binary_name.lower()}"):
            candidates.append(name)

    if not candidates:
        return None

    candidates.sort(key=len)
    return candidates[0]


def find_ffmpeg_member(names: Iterable[str]) -> Optional[str]:
    return find_binary_member(names, "ffmpeg.exe")


def is_windows() -> bool:
    return os.name == "nt"


def check_ffmpeg_exists(paths: AppPaths) -> bool:
    return paths.ffmpeg_path.exists() and paths.ffprobe_path.exists()


def missing_binaries(paths: AppPaths) -> dict[str, Path]:
    targets = {
        "ffmpeg.exe": paths.ffmpeg_path,
        "ffprobe.exe": paths.ffprobe_path,
    }
    return {name: path for name, path in targets.items() if not path.exists()}


def download_file(url: str, dest: Path, progress: ProgressCallback = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    if progress:
        progress(f"Download start: {url}")

    with urllib.request.urlopen(url, timeout=60) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header and total_header.isdigit() else 0
        downloaded = 0
        last_report = 0

        with open(tmp, "wb") as fp:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fp.write(chunk)
                downloaded += len(chunk)

                if progress and total > 0:
                    percent = int(downloaded * 100 / total)
                    if percent >= last_report + 5:
                        last_report = percent
                        progress(f"Download progress: {percent}%")

    tmp.replace(dest)

    if progress:
        progress(f"Download finished: {dest}")


def extract_binary_exe(zip_path: Path, binary_name: str, dest_path: Path, progress: ProgressCallback = None) -> None:
    if progress:
        progress(f"Extract {binary_name}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        member = find_binary_member(zf.namelist(), binary_name)
        if member is None:
            raise FfmpegDownloadError(f"{binary_name} was not found in zip.")

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_exe = dest_path.with_suffix(".exe.tmp")

        with zf.open(member, "r") as src, open(tmp_exe, "wb") as dst:
            shutil.copyfileobj(src, dst)

        tmp_exe.replace(dest_path)

    if progress:
        progress(f"Installed: {dest_path}")


def extract_ffmpeg_exe(zip_path: Path, ffmpeg_path: Path, progress: ProgressCallback = None) -> None:
    extract_binary_exe(zip_path, "ffmpeg.exe", ffmpeg_path, progress)


def verify_binary_basic(binary_path: Path, binary_name: str) -> None:
    try:
        result = subprocess.run(
            [str(binary_path), "-hide_banner", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FfmpegDownloadError(f"{binary_name} exists, but it could not run: {exc}") from exc
    if result.returncode != 0:
        raise FfmpegDownloadError(f"{binary_name} exists, but it did not run.")


def verify_ffmpeg_basic(ffmpeg_path: Path) -> None:
    verify_binary_basic(ffmpeg_path, "ffmpeg.exe")


def verify_ffprobe_basic(ffprobe_path: Path) -> None:
    verify_binary_basic(ffprobe_path, "ffprobe.exe")


def verify_nvenc(ffmpeg_path: Path) -> bool:
    result = subprocess.run(
        [str(ffmpeg_path), "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return False
    out = result.stdout.lower()
    return "hevc_nvenc" in out or "h264_nvenc" in out or "av1_nvenc" in out


def list_ffmpeg_encoders(ffmpeg_path: Path, timeout: int = 30) -> set[str]:
    try:
        result = subprocess.run(
            [str(ffmpeg_path), "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if result.returncode != 0:
        return set()

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        if name in COMMON_ENCODERS or name.endswith(("_nvenc", "_qsv", "_amf")) or name.startswith("libx26"):
            encoders.add(name)
    return encoders


def encoder_help_text(ffmpeg_path: Path, encoder: str, timeout: int = 20) -> str:
    try:
        result = subprocess.run(
            [str(ffmpeg_path), "-hide_banner", "-h", f"encoder={encoder}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def encoder_supports_option(help_text: str, option: str) -> bool:
    return re.search(rf"(^|\s)-{re.escape(option)}(\s|$)", help_text, flags=re.MULTILINE) is not None


def split_encode_modes_from_help(help_text: str) -> list[str]:
    if not encoder_supports_option(help_text, "split_encode_mode"):
        return []
    values: list[str] = []
    capturing = False
    for line in help_text.splitlines():
        if "-split_encode_mode" in line:
            capturing = True
            continue
        if not capturing:
            continue
        if re.match(r"\s+-[A-Za-z0-9_]", line):
            break
        match = re.match(r"\s+([A-Za-z0-9][A-Za-z0-9_-]*)\s+[-+]?\d+(?:\s|$)", line)
        if not match:
            continue
        candidate = match.group(1).lower()
        if candidate not in values:
            values.append(candidate)
    if not values:
        values.append("auto")
    elif "auto" not in values:
        values.insert(0, "auto")
    return values


def encoder_capabilities(ffmpeg_path: Path) -> dict[str, dict[str, object]]:
    capabilities: dict[str, dict[str, object]] = {}
    for encoder in sorted(list_ffmpeg_encoders(ffmpeg_path)):
        help_text = encoder_help_text(ffmpeg_path, encoder)
        capabilities[encoder] = {
            "encoder": encoder,
            "available": True,
            "split_encode_modes": split_encode_modes_from_help(help_text) if is_nvenc_codec(encoder) else [],
            "supports_split_encode_mode": encoder_supports_option(help_text, "split_encode_mode"),
        }
    return capabilities


def smoke_test_encoder(
    ffmpeg_path: Path,
    encoder: str,
    resource_id: str = "",
    split_encode_mode: str = "",
    timeout: int = 30,
) -> tuple[bool, str]:
    sink = "NUL" if os.name == "nt" else "/dev/null"
    command = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=320x240:rate=30:duration=1",
        "-frames:v",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        encoder,
    ]
    if is_nvenc_codec(encoder):
        command += ["-gpu", str(resource_index(resource_id))]
        if (
            split_encode_mode
            and split_encode_mode not in {"auto", "default"}
            and encoder in {"hevc_nvenc", "av1_nvenc"}
        ):
            command += ["-split_encode_mode", split_encode_mode]
    command += ["-f", "null", sink]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if result.returncode != 0:
        return False, (result.stdout or "").strip() or f"exit {result.returncode}"
    return True, ""


def ensure_ffmpeg_available(
    paths: AppPaths,
    auto_download: bool,
    url: str = GYAN_FFMPEG_ZIP_URL,
    progress: ProgressCallback = None,
) -> bool:
    ensure_dirs(paths)

    if check_ffmpeg_exists(paths):
        verify_ffmpeg_basic(paths.ffmpeg_path)
        verify_ffprobe_basic(paths.ffprobe_path)
        return True

    if not auto_download:
        return False

    if not is_windows():
        raise FfmpegDownloadError("Auto download is supported only on Windows.")

    zip_path = paths.download_dir / "ffmpeg-release-essentials.zip"
    missing = missing_binaries(paths)
    download_file(url, zip_path, progress)
    for binary_name, dest_path in missing.items():
        extract_binary_exe(zip_path, binary_name, dest_path, progress)
    verify_ffmpeg_basic(paths.ffmpeg_path)
    verify_ffprobe_basic(paths.ffprobe_path)

    if progress:
        if verify_nvenc(paths.ffmpeg_path):
            progress("NVENC encoder was found in ffmpeg.")
        else:
            progress("Warning: NVENC encoder was not found in ffmpeg.")

    return True
