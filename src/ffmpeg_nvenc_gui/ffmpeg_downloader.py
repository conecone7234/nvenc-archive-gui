from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Optional

from .core import AppPaths, ensure_dirs

ProgressCallback = Optional[Callable[[str], None]]

GYAN_FFMPEG_ZIP_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


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
