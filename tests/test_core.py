import json
import sys
import threading
from pathlib import Path

import ffmpeg_nvenc_gui.app as app_module
import ffmpeg_nvenc_gui.ffmpeg_downloader as downloader
from ffmpeg_nvenc_gui.core import (
    BACKEND_CPU,
    BACKEND_NVENC,
    CPU_RESOURCE_ID,
    EncodeProfile,
    GpuInfo,
    HardwareResource,
    JobSpec,
    OutputVariant,
    build_audio_command,
    build_concat_command,
    build_ffmpeg_command,
    build_job_specs,
    build_mux_command,
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
    parse_ffmpeg_args,
    probe_has_audio,
    profile_archive_dir,
    profile_from_state,
    profile_input_dir,
    profile_output_dir,
    profile_to_dict,
    resumable_specs,
    save_state,
    scan_profile_files,
    segment_dir_for,
    segment_file_name,
    segment_ranges,
    variant_resource_ids,
    write_concat_file,
)
from ffmpeg_nvenc_gui.app import EncoderApp, RuntimeJob
from ffmpeg_nvenc_gui.ffmpeg_downloader import (
    FfmpegDownloadError,
    find_binary_member,
    find_ffmpeg_member,
    missing_binaries,
    split_encode_modes_from_help,
    verify_ffmpeg_basic,
    verify_ffprobe_basic,
)


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
    profile.cpu_preset = "slow"
    profile.cpu_tune = "film"
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
    assert "-preset:v" in cmd
    assert "slow" in cmd
    assert "-tune:v" in cmd
    assert "film" in cmd
    assert "-crf" in cmd
    assert "-cq:v" not in cmd
    assert "-b:v" not in cmd


def test_output_profile_overrides_video_segment_command_and_extra_args(tmp_path: Path):
    profile = make_profile(tmp_path)
    variant = profile.outputs[1]
    variant.use_gpu = False
    variant.cpu_codec = "libx265"
    variant.cpu_preset = "slow"
    variant.cpu_tune = "grain"
    variant.rate_mode = "VBR"
    variant.bitrate = "8000k"
    variant.maxrate = "12000k"
    variant.bufsize = "24000k"
    variant.extra_input_args = "-noautorotate"
    variant.extra_video_args = "-g 120"
    variant.extra_output_args = "-movflags +frag_keyframe+empty_moov -map_metadata 0"

    cmd = build_ffmpeg_command(
        tmp_path / "ffmpeg.exe",
        tmp_path / "input.mkv",
        tmp_path / "chunk.mp4",
        profile,
        variant,
    )

    assert "libx265" in cmd
    assert cmd[cmd.index("-preset:v") + 1] == "slow"
    assert cmd[cmd.index("-tune:v") + 1] == "grain"
    assert ["-map", "0:v:0"] == cmd[cmd.index("-map") : cmd.index("-map") + 2]
    assert "-an" in cmd
    assert "-c:a" not in cmd
    assert "-noautorotate" in cmd
    assert "-g" in cmd
    assert "120" in cmd
    assert cmd.count("-movflags") == 1
    assert "+frag_keyframe+empty_moov" in cmd
    assert "+faststart" not in cmd


def test_parse_ffmpeg_args_preserves_windows_paths_and_removes_quotes():
    args = parse_ffmpeg_args(r'-metadata title="My Clip" -passlogfile "C:\temp\ffmpeg pass.log"')

    assert args == ["-metadata", "title=My Clip", "-passlogfile", r"C:\temp\ffmpeg pass.log"]


def test_audio_and_mux_commands_are_separate_from_video_segments(tmp_path: Path):
    profile = make_profile(tmp_path)
    variant = profile.outputs[0]
    variant.audio_codec = "aac"
    variant.audio_bitrate = "192k"
    variant.extra_audio_args = "-ar 48000"
    variant.extra_mux_args = "-map_metadata 0"

    audio_cmd = build_audio_command(
        tmp_path / "ffmpeg.exe",
        tmp_path / "input.mkv",
        tmp_path / "audio.m4a",
        profile,
        variant,
    )
    mux_cmd = build_mux_command(
        tmp_path / "ffmpeg.exe",
        tmp_path / "video.mp4",
        tmp_path / "audio.m4a",
        tmp_path / "output.mp4",
        profile,
        variant,
    )

    assert ["-map", "0:a:0"] == audio_cmd[audio_cmd.index("-map") : audio_cmd.index("-map") + 2]
    assert "-vn" in audio_cmd
    assert ["-c:a", "aac"] == audio_cmd[audio_cmd.index("-c:a") : audio_cmd.index("-c:a") + 2]
    assert ["-b:a", "192k"] == audio_cmd[audio_cmd.index("-b:a") : audio_cmd.index("-b:a") + 2]
    assert "-ar" in audio_cmd
    assert "48000" in audio_cmd
    assert ["-map", "0:v:0"] == mux_cmd[mux_cmd.index("-map") : mux_cmd.index("-map") + 2]
    second_map = mux_cmd.index("-map", mux_cmd.index("-map") + 1)
    assert ["-map", "1:a:0"] == mux_cmd[second_map : second_map + 2]
    assert ["-c", "copy"] == mux_cmd[mux_cmd.index("-c") : mux_cmd.index("-c") + 2]
    assert "-shortest" in mux_cmd
    assert "-map_metadata" in mux_cmd


def test_probe_has_audio_handles_missing_and_stream_results(tmp_path: Path, monkeypatch):
    src = tmp_path / "video.mp4"
    ffprobe = tmp_path / "ffprobe.exe"
    ffprobe.write_text("", encoding="utf-8")

    assert probe_has_audio(tmp_path / "missing-ffprobe.exe", src) is False

    class Result:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    monkeypatch.setattr("ffmpeg_nvenc_gui.core.subprocess.run", lambda *_args, **_kwargs: Result(""))

    assert probe_has_audio(ffprobe, src) is False

    monkeypatch.setattr("ffmpeg_nvenc_gui.core.subprocess.run", lambda *_args, **_kwargs: Result("0\n"))

    assert probe_has_audio(ffprobe, src) is True


def test_probe_has_audio_fails_fast_on_probe_errors(tmp_path: Path, monkeypatch):
    src = tmp_path / "video.mp4"
    ffprobe = tmp_path / "ffprobe.exe"
    ffprobe.write_text("", encoding="utf-8")

    class Result:
        returncode = 1
        stdout = ""
        stderr = "invalid input"

    monkeypatch.setattr("ffmpeg_nvenc_gui.core.subprocess.run", lambda *_args, **_kwargs: Result())

    try:
        probe_has_audio(ffprobe, src)
    except RuntimeError as exc:
        assert "exit 1" in str(exc)
        assert "invalid input" in str(exc)
    else:
        raise AssertionError("expected ffprobe return code failure to raise")

    def raise_probe_error(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr("ffmpeg_nvenc_gui.core.subprocess.run", raise_probe_error)

    try:
        probe_has_audio(ffprobe, src)
    except RuntimeError as exc:
        assert "permission denied" in str(exc)
    else:
        raise AssertionError("expected ffprobe invocation failure to raise")


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

    assert duplicate_output_targets(profile) == ["review/{source}.mp4"]

    profile.outputs[1].filename_template = "{source}-mobile"
    assert duplicate_output_targets(profile) == []


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
    assert specs == [JobSpec(src=str(src), profile_id="profile", variant_id="master", assigned_resource_id="nvidia:0")]


def test_build_job_specs_round_robins_encode_set_resources(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.hardware_resources = [
        HardwareResource(id=CPU_RESOURCE_ID, label="CPU", kind="cpu", backend=BACKEND_CPU),
        HardwareResource(id="nvidia:0", label="GPU 0", kind="gpu", backend=BACKEND_NVENC, concurrency_slots=2),
        HardwareResource(id="nvidia:1", label="GPU 1", kind="gpu", backend=BACKEND_NVENC, concurrency_slots=1),
    ]
    profile.resource_ids = ["nvidia:0", "nvidia:1"]
    profile.outputs[0].backend = BACKEND_NVENC
    profile.outputs[0].ffmpeg_encoder = "hevc_nvenc"
    profile.outputs[0].resource_ids = ["nvidia:0", "nvidia:1"]
    profile.outputs[1].enabled = False

    input_dir = Path(profile.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    sources = [input_dir / f"video-{index}.mkv" for index in range(3)]
    for src in sources:
        src.write_bytes(b"dummy")

    specs = build_job_specs(profile, sources)

    assert [spec.assigned_resource_id for spec in specs] == ["nvidia:0", "nvidia:1", "nvidia:0"]


def test_build_ffmpeg_command_uses_assigned_nvenc_resource(tmp_path: Path):
    profile = make_profile(tmp_path)
    variant = profile.outputs[0]
    variant.backend = BACKEND_NVENC
    variant.ffmpeg_encoder = "hevc_nvenc"
    variant.resource_ids = ["nvidia:2"]

    cmd = build_ffmpeg_command(
        tmp_path / "ffmpeg.exe",
        tmp_path / "input.mkv",
        tmp_path / "chunk.mp4",
        profile,
        variant,
        resource_id="nvidia:2",
    )

    assert cmd[cmd.index("-gpu") + 1] == "2"


def test_profile_to_dict_omits_deprecated_profile_defaults(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.hardware_resources = [
        HardwareResource(id=CPU_RESOURCE_ID, label="CPU", kind="cpu", backend=BACKEND_CPU),
        HardwareResource(id="nvidia:0", label="GPU 0", kind="gpu", backend=BACKEND_NVENC, concurrency_slots=2),
    ]
    profile.resource_ids = ["nvidia:0"]
    profile.outputs[0].backend = BACKEND_NVENC
    profile.outputs[0].ffmpeg_encoder = "hevc_nvenc"
    profile.outputs[0].resource_ids = ["nvidia:0"]

    data = profile_to_dict(profile)

    assert "codec" not in data
    assert "rate_mode" not in data
    assert data["hardware_resources"][1]["concurrency_slots"] == 2
    assert data["outputs"][0]["backend"] == BACKEND_NVENC
    assert data["outputs"][0]["ffmpeg_encoder"] == "hevc_nvenc"
    assert "use_gpu" not in data["outputs"][0]


def test_variant_resources_do_not_fallback_to_disabled_resource(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.hardware_resources = [
        HardwareResource(id=CPU_RESOURCE_ID, label="CPU", kind="cpu", backend=BACKEND_CPU),
        HardwareResource(id="nvidia:0", label="GPU 0", kind="gpu", backend=BACKEND_NVENC),
    ]
    profile.resource_ids = [CPU_RESOURCE_ID]
    profile.outputs[0].backend = BACKEND_NVENC
    profile.outputs[0].ffmpeg_encoder = "hevc_nvenc"
    profile.outputs[0].resource_ids = []

    assert variant_resource_ids(profile, profile.outputs[0]) == []


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


def test_output_filename_template_uses_source_and_output_profile_tokens(tmp_path: Path):
    profile = make_profile(tmp_path)
    variant = profile.outputs[0]
    variant.filename_template = "{source}-{profile}-{height}"

    output = output_path_for(profile, Path(profile.input_dir) / "Clip 01.mkv", variant)

    assert output.name == "Clip 01-Master 2160p-2160.mp4"


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
    explicit_cpu_profile = EncodeProfile.from_dict(
        {"use_gpu": False},
        paths,
        [GpuInfo(index=2, name="RTX Test")],
    )
    custom_cpu_profile = EncodeProfile.from_dict(
        {"use_gpu": False, "max_parallel_jobs": 3, "cq_value": 20, "bitrate": "6000k"},
        paths,
        [GpuInfo(index=2, name="RTX Test")],
    )
    stale_gpu_profile = EncodeProfile.from_dict(
        {"use_gpu": True, "gpu_index": 9, "gpu_name": "Old GPU"},
        paths,
        [GpuInfo(index=2, name="RTX Test")],
    )
    synced_gpu_profile = EncodeProfile.from_dict(
        {"use_gpu": True, "gpu_index": 2, "gpu_name": "Old GPU"},
        paths,
        [GpuInfo(index=2, name="RTX Test")],
    )

    assert Path(gpu_profile.input_dir) == paths.base_dir / "Incoming"
    assert gpu_profile.use_gpu is True
    assert gpu_profile.gpu_index == 2
    assert gpu_profile.gpu_name == "RTX Test"
    assert cpu_profile.use_gpu is False
    assert cpu_profile.max_parallel_jobs == 1
    assert cpu_profile.cq_value == 23
    assert cpu_profile.bitrate == "8000k"
    assert cpu_profile.cpu_preset == "medium"
    assert cpu_profile.cpu_tune == "none"
    assert explicit_cpu_profile.use_gpu is False
    assert explicit_cpu_profile.max_parallel_jobs == 1
    assert explicit_cpu_profile.cq_value == 23
    assert explicit_cpu_profile.bitrate == "8000k"
    assert custom_cpu_profile.max_parallel_jobs == 3
    assert custom_cpu_profile.cq_value == 20
    assert custom_cpu_profile.bitrate == "6000k"
    assert stale_gpu_profile.use_gpu is False
    assert stale_gpu_profile.gpu_index == 0
    assert stale_gpu_profile.gpu_name == ""
    assert stale_gpu_profile.max_parallel_jobs == 1
    assert stale_gpu_profile.cq_value == 23
    assert synced_gpu_profile.use_gpu is True
    assert synced_gpu_profile.gpu_name == "RTX Test"


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

    paths.state_file.write_bytes(b"\xff\xfe{not utf8")
    assert load_state(paths) is None


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


def test_segment_file_name_normalizes_container_and_partial_marker():
    variant = OutputVariant(
        id="review",
        name="Review",
        folder_name="review",
        container=".MKV",
    )

    assert segment_file_name(variant, 3) == "segment-00003.mkv"
    assert segment_file_name(variant, 3, partial=True) == "segment-00003.partial.mkv"
    assert segment_file_name(variant, 0, src=Path("A.mp4")) == "A-001.mkv"
    assert segment_file_name(variant, 1, partial=True, src=Path("A.mp4")) == "A-002.partial.mkv"
    long_name = "A" * 120
    assert segment_file_name(variant, 0, src=Path(f"{long_name}.mp4")) == f"{'A' * 40}-001.mkv"


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


def test_encoder_app_validate_profile_requires_enabled_output(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    profile = make_profile(tmp_path)
    for variant in profile.outputs:
        variant.enabled = False
    messages = []

    original_showerror = app_module.messagebox.showerror

    def showerror(title, message):
        messages.append((title, message))

    app_module.messagebox.showerror = showerror
    try:
        assert app.validate_profile_before_run(profile) is False
    finally:
        app_module.messagebox.showerror = original_showerror

    assert any("有効" in message for _, message in messages)


def test_encoder_app_output_cq_validation_depends_on_rate_mode():
    assert EncoderApp._cq_value_for_rate_mode("ABR", "", 18) == 18
    assert EncoderApp._cq_value_for_rate_mode("CBR", "not-number", 22) == 22
    assert EncoderApp._cq_value_for_rate_mode("CQ", "19", 18) == 19
    assert EncoderApp._cq_value_for_rate_mode("VBR", "20", 18) == 20
    assert EncoderApp._cq_value_for_rate_mode("CQ", "not-number", 18) is None


def test_encoder_app_output_cq_fallback_prefers_existing_then_form_value(tmp_path: Path):
    class Value:
        def __init__(self, value: str):
            self.value = value

        def get(self) -> str:
            return self.value

    app = EncoderApp.__new__(EncoderApp)
    profile = make_profile(tmp_path)
    app.current_profile = lambda: profile
    app.cq_var = Value("21")
    app.selected_output_id = None
    app.editing_outputs = [OutputVariant(id="existing", name="Existing", folder_name="existing", cq_value=25)]

    assert app._output_cq_fallback() == 21

    app.selected_output_id = "existing"
    assert app._output_cq_fallback() == 25

    app.selected_output_id = None
    app.cq_var = Value("not-number")
    assert app._output_cq_fallback() == profile.cq_value


def test_encoder_app_output_rate_controls_match_selected_mode():
    class Value:
        def __init__(self, value: str):
            self.value = value

        def get(self) -> str:
            return self.value

    class Widget:
        def __init__(self):
            self.state = None

        def configure(self, **kwargs):
            self.state = kwargs["state"]

    app = EncoderApp.__new__(EncoderApp)
    app.output_rate_mode_var = Value("ABR")
    app.output_cq_entry = Widget()
    app.output_bitrate_entry = Widget()
    app.output_maxrate_entry = Widget()
    app.output_bufsize_entry = Widget()

    app.update_output_rate_controls()

    assert app.output_cq_entry.state == app_module.tk.DISABLED
    assert app.output_bitrate_entry.state == app_module.tk.NORMAL
    assert app.output_maxrate_entry.state == app_module.tk.DISABLED
    assert app.output_bufsize_entry.state == app_module.tk.DISABLED

    app.output_rate_mode_var = Value("VBR")
    app.update_output_rate_controls()

    assert app.output_cq_entry.state == app_module.tk.NORMAL
    assert app.output_bitrate_entry.state == app_module.tk.NORMAL
    assert app.output_maxrate_entry.state == app_module.tk.NORMAL
    assert app.output_bufsize_entry.state == app_module.tk.NORMAL


def test_validate_encoder_capabilities_ignores_stale_sfe_for_h264_nvenc(tmp_path: Path, monkeypatch):
    app = EncoderApp.__new__(EncoderApp)
    app.encoder_capabilities = {
        "h264_nvenc": {
            "available": True,
            "supports_split_encode_mode": False,
            "split_encode_modes": [],
        }
    }
    app.paths = build_paths(tmp_path)
    app.log = lambda _text: None
    errors = []
    smoke_calls = []

    profile = make_profile(tmp_path)
    variant = profile.outputs[0]
    variant.backend = "nvenc"
    variant.ffmpeg_encoder = "h264_nvenc"
    variant.split_encode_mode = "2"
    spec = JobSpec(src=str(tmp_path / "input.mkv"), profile_id=profile.id, variant_id=variant.id, assigned_resource_id="nvidia:0")

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    def fake_smoke(ffmpeg_path, encoder, resource_id="", split_encode_mode="", timeout=30):
        smoke_calls.append((encoder, resource_id, split_encode_mode))
        return True, ""

    monkeypatch.setattr(app_module, "smoke_test_encoder", fake_smoke)

    assert app.validate_encoder_capabilities_before_run(profile, [spec]) is True
    assert errors == []
    assert smoke_calls == [("h264_nvenc", "nvidia:0", "")]


def test_validate_encoder_capabilities_rejects_invalid_sfe_for_hevc_nvenc(tmp_path: Path, monkeypatch):
    app = EncoderApp.__new__(EncoderApp)
    app.encoder_capabilities = {
        "hevc_nvenc": {
            "available": True,
            "supports_split_encode_mode": True,
            "split_encode_modes": ["auto", "disabled"],
        }
    }
    app.paths = build_paths(tmp_path)
    app.log = lambda _text: None
    errors = []
    smoke_calls = []

    profile = make_profile(tmp_path)
    variant = profile.outputs[0]
    variant.backend = "nvenc"
    variant.ffmpeg_encoder = "hevc_nvenc"
    variant.split_encode_mode = "2"
    spec = JobSpec(src=str(tmp_path / "input.mkv"), profile_id=profile.id, variant_id=variant.id, assigned_resource_id="nvidia:0")

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))
    monkeypatch.setattr(app_module, "smoke_test_encoder", lambda *_args, **_kwargs: smoke_calls.append(_args) or (True, ""))

    assert app.validate_encoder_capabilities_before_run(profile, [spec]) is False
    assert smoke_calls == []
    assert any("split_encode_mode=2" in message for _title, message in errors)


def test_validate_encoder_capabilities_allows_disabled_sfe_when_option_exists(tmp_path: Path, monkeypatch):
    app = EncoderApp.__new__(EncoderApp)
    app.encoder_capabilities = {
        "hevc_nvenc": {
            "available": True,
            "supports_split_encode_mode": True,
            "split_encode_modes": [],
        }
    }
    app.paths = build_paths(tmp_path)
    app.log = lambda _text: None
    errors = []
    smoke_calls = []

    profile = make_profile(tmp_path)
    variant = profile.outputs[0]
    variant.backend = "nvenc"
    variant.ffmpeg_encoder = "hevc_nvenc"
    variant.split_encode_mode = "disabled"
    spec = JobSpec(src=str(tmp_path / "input.mkv"), profile_id=profile.id, variant_id=variant.id, assigned_resource_id="nvidia:0")

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    def fake_smoke(ffmpeg_path, encoder, resource_id="", split_encode_mode="", timeout=30):
        smoke_calls.append((encoder, resource_id, split_encode_mode))
        return True, ""

    monkeypatch.setattr(app_module, "smoke_test_encoder", fake_smoke)

    assert app.split_encode_modes_for_encoder("hevc_nvenc") == ["auto", "disabled"]
    assert app.validate_encoder_capabilities_before_run(profile, [spec]) is True
    assert errors == []
    assert smoke_calls == [("hevc_nvenc", "nvidia:0", "disabled")]


def test_validate_profile_before_run_rejects_output_without_enabled_resource(tmp_path: Path, monkeypatch):
    app = EncoderApp.__new__(EncoderApp)
    profile = make_profile(tmp_path)
    profile.hardware_resources = [
        HardwareResource(id=CPU_RESOURCE_ID, label="CPU", kind="cpu", backend=BACKEND_CPU),
        HardwareResource(id="nvidia:0", label="GPU 0", kind="gpu", backend=BACKEND_NVENC),
    ]
    profile.resource_ids = [CPU_RESOURCE_ID]
    profile.outputs[0].backend = BACKEND_NVENC
    profile.outputs[0].ffmpeg_encoder = "hevc_nvenc"
    profile.outputs[0].resource_ids = []
    profile.outputs[1].enabled = False
    errors = []

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    assert app.validate_profile_before_run(profile) is False
    assert any("使用可能なリソース" in message for _title, message in errors)


def test_reserve_resource_slots_uses_readable_status(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.active_resource_slots = {}
    app.active_resource_slot_indexes = {}
    app.active_jobs = {}
    profile = make_profile(tmp_path)
    job = RuntimeJob(
        job_id=1,
        spec=JobSpec(src=str(tmp_path / "input.mkv"), profile_id=profile.id, variant_id=profile.outputs[0].id),
        profile=profile,
        variant=profile.outputs[0],
        tmp_out=tmp_path / "tmp.mp4",
        out_file=tmp_path / "out.mp4",
        log_file=tmp_path / "job.log",
        resource_id="nvidia:0",
    )

    assert app.reserve_resource_slots(job, {"nvidia:0": 2}) is True
    assert job.status == "準備中"
    assert app.active_resource_slots == {"nvidia:0": 2}
    assert job.resource_slot_indexes == [0, 1]


def test_resource_slot_indices_do_not_overlap_after_release(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.lock = threading.Lock()
    app.active_resource_slots = {}
    app.active_resource_slot_indexes = {}
    app.active_jobs = {}
    profile = make_profile(tmp_path)
    profile.max_parallel_jobs = 1

    def make_job(job_id: int) -> RuntimeJob:
        return RuntimeJob(
            job_id=job_id,
            spec=JobSpec(src=str(tmp_path / f"input-{job_id}.mkv"), profile_id=profile.id, variant_id=profile.outputs[0].id),
            profile=profile,
            variant=profile.outputs[0],
            tmp_out=tmp_path / f"tmp-{job_id}.mp4",
            out_file=tmp_path / f"out-{job_id}.mp4",
            log_file=tmp_path / f"job-{job_id}.log",
            resource_id="nvidia:0",
        )

    first = make_job(1)
    second = make_job(2)
    third = make_job(3)

    assert app.reserve_resource_slots(first, {"nvidia:0": 3}) is True
    assert app.reserve_resource_slots(second, {"nvidia:0": 3}) is True
    assert first.resource_slot_indexes == [0]
    assert second.resource_slot_indexes == [1]

    app.release_resource_slots(first)
    assert app.active_resource_slot_indexes == {"nvidia:0": {1}}

    assert app.reserve_resource_slots(third, {"nvidia:0": 3}) is True
    assert third.resource_slot_indexes == [0]
    assert set(third.resource_slot_indexes).isdisjoint(second.resource_slot_indexes)


def test_run_process_publishes_process_under_lock_and_honors_stop(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.lock = threading.Lock()
    app.stop_requested = True
    app.log = lambda _text: None
    profile = make_profile(tmp_path)
    job = RuntimeJob(
        job_id=1,
        spec=JobSpec(src=str(tmp_path / "input.mkv"), profile_id=profile.id, variant_id=profile.outputs[0].id),
        profile=profile,
        variant=profile.outputs[0],
        tmp_out=tmp_path / "tmp.mp4",
        out_file=tmp_path / "out.mp4",
        log_file=tmp_path / "job.log",
    )

    with open(job.log_file, "w", encoding="utf-8") as log_fp:
        ret = app.run_process(job, [sys.executable, "-c", "import time; time.sleep(5)"], log_fp, None)

    assert ret != 0
    assert job.process is None


def test_skip_audio_output_logs_reason_and_moves_joined_video(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.lock = threading.Lock()
    log_messages = []
    app.log = log_messages.append
    profile = make_profile(tmp_path)
    job = RuntimeJob(
        job_id=1,
        spec=JobSpec(src=str(tmp_path / "input.mkv"), profile_id=profile.id, variant_id=profile.outputs[0].id),
        profile=profile,
        variant=profile.outputs[0],
        tmp_out=tmp_path / "tmp.mp4",
        out_file=tmp_path / "out.mp4",
        log_file=tmp_path / "job.log",
    )
    joined_video = tmp_path / "joined.mp4"
    joined_video.write_bytes(b"video")

    with open(job.log_file, "w", encoding="utf-8") as log_fp:
        app.skip_audio_output(job, Path(job.spec.src), joined_video, log_fp, "No audio stream detected")

    assert job.tmp_out.read_bytes() == b"video"
    assert not joined_video.exists()
    assert job.status == "映像のみ"
    assert job.message == "video-only"
    assert any("No audio stream detected" in message for message in log_messages)
    assert "Audio skipped: No audio stream detected" in job.log_file.read_text(encoding="utf-8")


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


def test_split_encode_modes_parse_only_option_values():
    help_text = """
Encoder hevc_nvenc [NVIDIA NVENC hevc encoder]:
  unrelated option mentions 2-pass and h264 text.
  -split_encode_mode <int> E..V....... Set split encoding mode
     auto            0            E..V.......
     disabled        1            E..V.......
     forced          2            E..V.......
     2               3            E..V.......
  -gpu <int>         E..V....... Selects which NVENC capable GPU to use
"""

    assert split_encode_modes_from_help(help_text) == ["auto", "disabled", "forced", "2"]
    assert split_encode_modes_from_help("h264 mentions 2 but no option") == []


def test_missing_binaries_preserves_existing_ffmpeg(tmp_path: Path):
    paths = build_paths(tmp_path)
    paths.ffmpeg_path.parent.mkdir(parents=True, exist_ok=True)
    paths.ffmpeg_path.write_bytes(b"custom ffmpeg")

    missing = missing_binaries(paths)

    assert "ffmpeg.exe" not in missing
    assert missing == {"ffprobe.exe": paths.ffprobe_path}


def test_verify_binaries_wrap_oserror():
    original_run = downloader.subprocess.run

    def raise_oserror(*args, **kwargs):
        raise OSError("permission denied")

    downloader.subprocess.run = raise_oserror
    try:
        for verify, binary_name in (
            (verify_ffmpeg_basic, "ffmpeg.exe"),
            (verify_ffprobe_basic, "ffprobe.exe"),
        ):
            try:
                verify(Path(binary_name))
            except FfmpegDownloadError as exc:
                text = str(exc)
                assert binary_name in text
                assert "permission denied" in text
            else:
                raise AssertionError(f"{binary_name} OSError was not wrapped")
    finally:
        downloader.subprocess.run = original_run


def test_encoder_app_filesystem_errors_show_messagebox(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.log_messages = []
    app.log = app.log_messages.append
    profile = make_profile(tmp_path)
    messages = []

    original_ensure = app_module.ensure_profile_dirs
    original_scan = app_module.scan_profile_files
    original_showerror = app_module.messagebox.showerror

    def showerror(title, message):
        messages.append((title, message))

    def raise_setup_error(profile):
        raise OSError("setup denied")

    def raise_scan_error(profile):
        raise OSError("scan denied")

    app_module.messagebox.showerror = showerror
    app_module.ensure_profile_dirs = raise_setup_error
    app_module.scan_profile_files = raise_scan_error
    try:
        assert app.ensure_profile_dirs_or_show_error(profile) is False
        assert app.scan_profile_files_or_show_error(profile) is None
        app.move_finished_sources(profile)
    finally:
        app_module.ensure_profile_dirs = original_ensure
        app_module.scan_profile_files = original_scan
        app_module.messagebox.showerror = original_showerror

    assert any("setup denied" in message for _, message in messages)
    assert any("scan denied" in message for _, message in messages)
    assert any("Move skipped" in message for message in app.log_messages)
