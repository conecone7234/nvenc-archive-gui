from pathlib import Path

from ffmpeg_nvenc_gui.core import (
    EncodeSettings,
    JobSpec,
    build_ffmpeg_command,
    build_job_specs,
    build_paths,
    clear_state,
    load_state,
    output_path_for,
    resumable_specs,
    save_state,
    scan_obs_files,
)
from ffmpeg_nvenc_gui.ffmpeg_downloader import find_ffmpeg_member


def test_build_cq_4k_command(tmp_path: Path):
    settings = EncodeSettings(codec="hevc_nvenc", rate_mode="CQ", cq_value=15, height_4k=2160)
    cmd = build_ffmpeg_command(
        tmp_path / "ffmpeg" / "ffmpeg.exe",
        tmp_path / "OBSData" / "input file.mkv",
        tmp_path / "tmp" / "input-4K.mp4",
        "4K",
        settings,
    )

    assert "hevc_nvenc" in cmd
    assert "-cq:v" in cmd
    assert "15" in cmd
    assert "scale=-1:2160:flags=lanczos+accurate_rnd" in cmd
    assert "-map" in cmd
    assert "0" in cmd


def test_build_vbr_mp4_command(tmp_path: Path):
    settings = EncodeSettings(rate_mode="VBR", bitrate="30000k", maxrate="45000k", bufsize="90000k", cq_value=18)
    cmd = build_ffmpeg_command(
        tmp_path / "ffmpeg.exe",
        tmp_path / "a.mkv",
        tmp_path / "a.mp4",
        "MP4",
        settings,
    )

    assert "-rc:v" in cmd
    assert "vbr" in cmd
    assert "-maxrate:v" in cmd
    assert "45000k" in cmd
    assert not any(str(x).startswith("scale=-1:") for x in cmd)


def test_scan_and_job_specs_skip_existing_outputs(tmp_path: Path):
    paths = build_paths(tmp_path)
    for directory in [paths.obs_dir, paths.mp4_dir, paths.k4_dir, paths.tmp_dir, paths.log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    src = paths.obs_dir / "video.mkv"
    src.write_bytes(b"dummy")
    output_path_for(paths, src, "MP4").write_bytes(b"done")

    files = scan_obs_files(paths)
    assert len(files) == 1
    assert files[0].label == "[missing 4K]"

    specs = build_job_specs(paths, [src], EncodeSettings(make_4k=True, make_mp4=True))
    assert specs == [JobSpec(src=str(src), mode="4K")]


def test_state_resume_filters_completed_jobs(tmp_path: Path):
    paths = build_paths(tmp_path)
    paths.tmp_dir.mkdir(parents=True, exist_ok=True)
    paths.k4_dir.mkdir(parents=True, exist_ok=True)
    paths.mp4_dir.mkdir(parents=True, exist_ok=True)

    src = paths.obs_dir / "video.mkv"
    specs = [JobSpec(src=str(src), mode="4K"), JobSpec(src=str(src), mode="MP4")]
    save_state(paths, EncodeSettings(), specs)

    output_path_for(paths, src, "4K").parent.mkdir(parents=True, exist_ok=True)
    output_path_for(paths, src, "4K").write_bytes(b"done")

    data = load_state(paths)
    assert data is not None
    resume = resumable_specs(paths, data)
    assert resume == [JobSpec(src=str(src), mode="MP4")]

    clear_state(paths)
    assert load_state(paths) is None


def test_find_ffmpeg_member():
    names = [
        "readme.txt",
        "ffmpeg-2026-essentials_build/bin/ffprobe.exe",
        "ffmpeg-2026-essentials_build/bin/ffmpeg.exe",
    ]
    assert find_ffmpeg_member(names) == "ffmpeg-2026-essentials_build/bin/ffmpeg.exe"
