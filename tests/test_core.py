import json
from pathlib import Path

from ffmpeg_nvenc_gui.core import (
    EncodeProfile,
    GpuInfo,
    JobSpec,
    OutputVariant,
    build_concat_command,
    build_ffmpeg_command,
    build_job_specs,
    build_paths,
    clear_state,
    duplicate_output_targets,
    ensure_profile_dirs,
    format_seconds,
    load_profiles,
    load_state,
    missing_profile_dirs,
    missing_rate_fields,
    normalize_container_extension,
    output_path_for,
    profile_archive_dir,
    profile_from_state,
    profile_input_dir,
    profile_output_dir,
    resumable_specs,
    save_state,
    scan_profile_files,
    segment_dir_for,
    segment_ranges,
    write_concat_file,
)
from ffmpeg_nvenc_gui.app import EncoderApp
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
    input_index = cmd.index("-i")
    assert cmd.index("-ss") < input_index
    assert cmd.index("-t") < input_index
    assert cmd[input_index + 1] == str(tmp_path / "Incoming" / "input file.mkv")
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


def test_faststart_is_only_used_for_mov_mp4_family(tmp_path: Path):
    profile = make_profile(tmp_path)
    mp4_variant = profile.outputs[0]
    mkv_variant = OutputVariant(
        id="matroska",
        name="Matroska",
        folder_name="matroska",
        height=None,
        container="mkv",
    )

    mp4_cmd = build_ffmpeg_command(
        tmp_path / "ffmpeg.exe",
        tmp_path / "a.mkv",
        tmp_path / "a.mp4",
        profile,
        mp4_variant,
    )
    mkv_cmd = build_ffmpeg_command(
        tmp_path / "ffmpeg.exe",
        tmp_path / "a.mkv",
        tmp_path / "a.mkv",
        profile,
        mkv_variant,
    )
    mp4_concat = build_concat_command(tmp_path / "ffmpeg.exe", tmp_path / "concat.txt", tmp_path / "final.mp4")
    mkv_concat = build_concat_command(tmp_path / "ffmpeg.exe", tmp_path / "concat.txt", tmp_path / "final.mkv")

    assert "-movflags" in mp4_cmd
    assert "-movflags" not in mkv_cmd
    assert "-movflags" in mp4_concat
    assert "-movflags" not in mkv_concat


def test_rate_modes_require_needed_bitrate_fields(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.rate_mode = "VBR"
    profile.bitrate = ""
    profile.maxrate = "40000k"
    profile.bufsize = "80000k"

    assert missing_rate_fields(profile) == ["bitrate"]

    try:
        build_ffmpeg_command(
            tmp_path / "ffmpeg.exe",
            tmp_path / "a.mkv",
            tmp_path / "a.mp4",
            profile,
            profile.outputs[0],
        )
    except ValueError as exc:
        assert "VBR requires: bitrate" in str(exc)
    else:
        raise AssertionError("expected missing bitrate to raise")


def test_empty_profile_dirs_are_required_before_file_scans(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.input_dir = ""

    assert missing_profile_dirs(profile) == ["input_dir"]
    assert scan_profile_files(profile) == []

    try:
        ensure_profile_dirs(profile)
    except ValueError as exc:
        assert "input_dir" in str(exc)
    else:
        raise AssertionError("expected empty input_dir to raise")


def test_duplicate_output_targets_are_detected(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.outputs = [
        OutputVariant(id="a", name="A", folder_name="review", height=1080, container="mp4"),
        OutputVariant(id="b", name="B", folder_name="review", height=720, container=".MP4"),
        OutputVariant(id="c", name="C", folder_name="review", height=720, container="mkv"),
    ]

    assert duplicate_output_targets(profile) == ["review.mp4"]


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


def test_output_variant_normalizes_safe_container_extensions(tmp_path: Path):
    profile = make_profile(tmp_path)
    variant = OutputVariant.from_dict(
        {
            "id": "archive",
            "name": "Archive MOV",
            "folder_name": "archive-mov",
            "container": ".MOV",
        }
    )
    unsafe = OutputVariant.from_dict(
        {
            "id": "unsafe",
            "name": "Unsafe",
            "folder_name": "unsafe",
            "container": "../outside",
        }
    )

    assert normalize_container_extension(" .MKV ") == "mkv"
    assert normalize_container_extension("../outside", default="") == ""
    assert variant.container == "mov"
    assert unsafe.container == "mp4"
    assert output_path_for(profile, Path(profile.input_dir) / "video.mkv", unsafe).name == "video.mp4"


def test_profile_and_variant_loading_ignore_unknown_keys_and_normalize_types(tmp_path: Path):
    paths = build_paths(tmp_path)
    profile = EncodeProfile.from_dict(
        {
            "id": " loaded ",
            "name": "Loaded",
            "max_parallel_jobs": "3",
            "segment_minutes": "12",
            "use_gpu": "false",
            "gpu_index": "2",
            "rate_mode": "vbr",
            "cq_value": "22",
            "unknown": "ignored",
            "outputs": [
                {
                    "id": "review",
                    "name": "Review",
                    "folder_name": "../Review Folder",
                    "height": "1080",
                    "container": ".MKV",
                    "enabled": "false",
                    "extra": "ignored",
                }
            ],
        },
        paths,
        [],
    )

    assert profile.id == "loaded"
    assert profile.max_parallel_jobs == 3
    assert profile.segment_minutes == 12
    assert profile.use_gpu is False
    assert profile.gpu_index == 2
    assert profile.rate_mode == "VBR"
    assert profile.cq_value == 22
    assert profile.outputs[0].folder_name == "Review-Folder"
    assert profile.outputs[0].height == 1080
    assert profile.outputs[0].container == "mkv"
    assert profile.outputs[0].enabled is False

    fallback = EncodeProfile.from_dict({"outputs": None}, paths, [])
    assert fallback.outputs

    direct = EncodeProfile(id="direct", name="Direct", input_dir="in", output_dir="out", archive_dir="arch", outputs=None)
    assert direct.outputs == []

    duplicate_ids = EncodeProfile.from_dict(
        {
            "outputs": [
                {"id": "same", "name": "A", "folder_name": "a"},
                {"id": "same", "name": "B", "folder_name": "b"},
            ]
        },
        paths,
        [],
    )
    assert duplicate_ids.outputs[0].id == "same"
    assert duplicate_ids.outputs[1].id != "same"


def test_encode_profile_from_dict_uses_supplied_paths_and_gpus(tmp_path: Path):
    paths = build_paths(tmp_path)

    gpu_profile = EncodeProfile.from_dict({}, paths, [GpuInfo(index=2, name="RTX Test")])
    cpu_profile = EncodeProfile.from_dict({}, paths, [])

    assert Path(gpu_profile.input_dir) == paths.base_dir / "Incoming"
    assert gpu_profile.use_gpu is True
    assert gpu_profile.gpu_index == 2
    assert gpu_profile.gpu_name == "RTX Test"
    assert cpu_profile.use_gpu is False
    assert cpu_profile.max_parallel_jobs == 1


def test_load_profiles_defaults_missing_fields_from_supplied_paths(tmp_path: Path):
    paths = build_paths(tmp_path)
    paths.config_file.write_text(
        json.dumps({"profiles": [{"id": "partial", "name": "Partial"}]}),
        encoding="utf-8",
    )

    profiles = load_profiles(paths, [])

    assert len(profiles) == 1
    assert Path(profiles[0].input_dir) == paths.base_dir / "Incoming"
    assert Path(profiles[0].output_dir) == paths.base_dir / "Encoded"
    assert profiles[0].use_gpu is False


def test_corrupt_json_files_fall_back_safely(tmp_path: Path):
    paths = build_paths(tmp_path)
    paths.config_file.parent.mkdir(parents=True, exist_ok=True)
    paths.config_file.write_text("{not json", encoding="utf-8")
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text("{not json", encoding="utf-8")

    profiles = load_profiles(paths, [])

    assert len(profiles) == 1
    assert profiles[0].id == "default"
    assert load_state(paths) is None

    paths.config_file.write_text(json.dumps({"profiles": {"bad": "shape"}}), encoding="utf-8")
    assert len(load_profiles(paths, [])) == 1


def test_ensure_profile_dirs_uses_resolved_profile_paths(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.input_dir = str(tmp_path / "base" / ".." / "incoming")
    profile.output_dir = str(tmp_path / "base" / ".." / "encoded")
    profile.archive_dir = str(tmp_path / "base" / ".." / "archive")

    ensure_profile_dirs(profile)

    assert profile_input_dir(profile).exists()
    assert profile_output_dir(profile).exists()
    assert profile_archive_dir(profile).exists()
    assert (profile_output_dir(profile) / profile.outputs[0].folder_name).exists()


def test_format_seconds_carries_rounded_milliseconds():
    assert format_seconds(1.9999) == "00:00:02.000"
    assert format_seconds(3599.9999) == "01:00:00.000"


def test_write_concat_file_escapes_single_quotes(tmp_path: Path):
    list_file = tmp_path / "concat.txt"
    segment = tmp_path / "clip'segment.mp4"

    write_concat_file(list_file, [segment])

    text = list_file.read_text(encoding="utf-8")
    assert "clip\\'segment.mp4" in text
    assert "clip'\\''segment" not in text


def test_segment_dir_includes_profile_and_encode_settings(tmp_path: Path):
    paths = build_paths(tmp_path)
    src = tmp_path / "Incoming" / "video.mkv"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"first")
    profile = make_profile(tmp_path)
    variant = profile.outputs[0]

    same_variant_other_profile = make_profile(tmp_path)
    same_variant_other_profile.id = "other-profile"
    edited_settings = make_profile(tmp_path)
    edited_settings.cq_value = profile.cq_value + 1

    base_dir = segment_dir_for(paths, src, profile, variant)
    other_profile_dir = segment_dir_for(paths, src, same_variant_other_profile, same_variant_other_profile.outputs[0])
    edited_settings_dir = segment_dir_for(paths, src, edited_settings, edited_settings.outputs[0])
    src.write_bytes(b"second source content")
    replaced_source_dir = segment_dir_for(paths, src, profile, variant)

    assert base_dir != other_profile_dir
    assert base_dir != edited_settings_dir
    assert base_dir != replaced_source_dir


def test_encoder_app_profile_helpers_use_ids_for_duplicates(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.profiles = [
        EncodeProfile(
            id="profile_alpha",
            name="Archive",
            input_dir=str(tmp_path / "in-a"),
            output_dir=str(tmp_path / "out-a"),
            archive_dir=str(tmp_path / "archive-a"),
        ),
        EncodeProfile(
            id="profile_beta",
            name="Archive",
            input_dir=str(tmp_path / "in-b"),
            output_dir=str(tmp_path / "out-b"),
            archive_dir=str(tmp_path / "archive-b"),
        ),
        EncodeProfile(
            id="profile_gamma",
            name="Profile 4",
            input_dir=str(tmp_path / "in-c"),
            output_dir=str(tmp_path / "out-c"),
            archive_dir=str(tmp_path / "archive-c"),
        ),
    ]

    labels = EncoderApp._profile_labels(app)

    assert labels == ["Archive (alpha)", "Archive (beta)", "Profile 4"]
    assert EncoderApp.profile_index_by_id(app, "profile_beta") == 1
    assert EncoderApp.has_duplicate_profile_name(app, "profile_alpha", "Archive") is True
    assert EncoderApp.has_duplicate_profile_name(app, "profile_alpha", "Unique") is False
    assert EncoderApp.unique_profile_name(app, "Profile") == "Profile 5"


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
    restored = profile_from_state(data, paths, [])
    resume = resumable_specs(restored, data)
    assert resume == [JobSpec(src=str(src), profile_id="profile", variant_id="review")]

    clear_state(paths)
    assert load_state(paths) is None


def test_resumable_specs_skips_legacy_state_jobs(tmp_path: Path):
    profile = make_profile(tmp_path)
    src = Path(profile.input_dir) / "video.mkv"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"dummy")

    data = {"version": 1, "jobs": [{"src": str(src), "mode": "4K"}]}

    assert resumable_specs(profile, data) == []


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
