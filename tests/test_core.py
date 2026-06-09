from pathlib import Path

from ffmpeg_nvenc_gui.core import (
    EncodeProfile,
    JobSpec,
    OutputVariant,
    build_ffmpeg_command,
    build_job_specs,
    build_paths,
    clear_state,
    load_state,
    output_path_for,
    profile_from_state,
    resumable_specs,
    save_state,
    scan_profile_files,
    segment_ranges,
)
from ffmpeg_nvenc_gui.ffmpeg_downloader import find_binary_member, find_ffmpeg_member


def make_profile(tmp_path: Path) -> EncodeProfile:
    return EncodeProfile(
        id="profile",
        name="Archive",
        input_dir=str(tmp_path / "Incoming"),
        output_dir=str(tmp_path / "Encoded"),
        archive_dir=str(tmp_path / "SourceArchive"),
        max_parallel_jobs=2,
        segment_minutes=10,
        use_gpu=True,
        gpu_index=0,
        codec="hevc_nvenc",
        rate_mode="CQ",
        cq_value=15,
        outputs=[
            OutputVariant(
                id="master",
                name="Master 2160p",
                folder_name="master-2160p",
                height=2160,
                container="mp4",
            ),
            OutputVariant(
                id="review",
                name="Review 1080p",
                folder_name="review-1080p",
                height=1080,
                container="mp4",
            ),
        ],
    )


def test_build_cq_gpu_command(tmp_path: Path):
    profile = make_profile(tmp_path)
    variant = profile.outputs[0]
    cmd = build_ffmpeg_command(
        tmp_path / "ffmpeg" / "ffmpeg.exe",
        tmp_path / "Incoming" / "input file.mkv",
        tmp_path / "tmp" / "input-master.mp4",
        profile,
        variant,
        start_seconds=60,
        duration_seconds=600,
    )

    assert "hevc_nvenc" in cmd
    assert "-gpu" in cmd
    assert "-cq:v" in cmd
    assert "15" in cmd
    assert "scale=-2:2160:flags=lanczos+accurate_rnd" in cmd
    assert "-ss" in cmd
    assert "-t" in cmd
    assert "-map" in cmd
    assert "0" in cmd


def test_build_cpu_cq_uses_crf_and_hides_bitrate(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.use_gpu = False
    profile.cpu_codec = "libx264"
    profile.rate_mode = "CQ"
    variant = profile.outputs[1]
    cmd = build_ffmpeg_command(
        tmp_path / "ffmpeg.exe",
        tmp_path / "a.mkv",
        tmp_path / "a.mp4",
        profile,
        variant,
    )

    assert "libx264" in cmd
    assert "yuv420p" in cmd
    assert "-crf" in cmd
    assert "-cq:v" not in cmd
    assert "-b:v" not in cmd


def test_scan_and_job_specs_skip_existing_outputs(tmp_path: Path):
    profile = make_profile(tmp_path)
    input_dir = Path(profile.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    src = input_dir / "video.mkv"
    src.write_bytes(b"dummy")
    output_path_for(profile, src, profile.outputs[1]).parent.mkdir(parents=True, exist_ok=True)
    output_path_for(profile, src, profile.outputs[1]).write_bytes(b"done")

    files = scan_profile_files(profile)
    assert len(files) == 1
    assert files[0].label(profile) == "残り 1/2"

    specs = build_job_specs(profile, [src])
    assert specs == [JobSpec(src=str(src), profile_id="profile", variant_id="master")]


def test_state_resume_filters_completed_jobs(tmp_path: Path):
    paths = build_paths(tmp_path)
    profile = make_profile(tmp_path)
    input_dir = Path(profile.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    src = input_dir / "video.mkv"
    src.write_bytes(b"dummy")
    specs = [
        JobSpec(src=str(src), profile_id="profile", variant_id="master"),
        JobSpec(src=str(src), profile_id="profile", variant_id="review"),
    ]
    save_state(paths, profile, specs)

    output_path_for(profile, src, profile.outputs[0]).parent.mkdir(parents=True, exist_ok=True)
    output_path_for(profile, src, profile.outputs[0]).write_bytes(b"done")

    data = load_state(paths)
    assert data is not None
    restored = profile_from_state(data)
    resume = resumable_specs(restored, data)
    assert resume == [JobSpec(src=str(src), profile_id="profile", variant_id="review")]

    clear_state(paths)
    assert load_state(paths) is None


def test_segment_ranges_split_by_duration():
    assert segment_ranges(125.0, 60) == [(0.0, 60.0), (60.0, 60.0), (120.0, 5.0)]
    assert segment_ranges(None, 60) == [(0.0, None)]


def test_find_ffmpeg_and_ffprobe_members():
    names = [
        "readme.txt",
        "ffmpeg-2026-essentials_build/bin/ffprobe.exe",
        "ffmpeg-2026-essentials_build/bin/ffmpeg.exe",
    ]
    assert find_ffmpeg_member(names) == "ffmpeg-2026-essentials_build/bin/ffmpeg.exe"
    assert find_binary_member(names, "ffprobe.exe") == "ffmpeg-2026-essentials_build/bin/ffprobe.exe"
