import inspect
import json
import sys
import threading
from pathlib import Path

import ffmpeg_nvenc_gui.app as app_module
import ffmpeg_nvenc_gui.core as core_module
import ffmpeg_nvenc_gui.ffmpeg_downloader as downloader
from ffmpeg_nvenc_gui.app import (
    EncoderApp,
    RuntimeJob,
    ScaleFlagsSelector,
    backend_accepts_rate_mode,
    default_output_backend_for_resource_ids,
    profile_uses_nvenc_resource,
    rate_modes_for_backend,
    select_compatible_resource_ids,
)
from ffmpeg_nvenc_gui.core import (
    BACKEND_AMF,
    BACKEND_CPU,
    BACKEND_NVENC,
    BACKEND_QSV,
    CPU_RESOURCE_ID,
    EncodeProfile,
    GpuInfo,
    HardwareResource,
    JobSpec,
    OutputVariant,
    audio_containers_for_codec,
    build_audio_command,
    build_concat_command,
    build_ffmpeg_command,
    build_job_specs,
    build_mux_command,
    build_paths,
    clear_cpu_resource_cache,
    clear_state,
    cpu_resources_from_wmi_data,
    detect_cpu_resources,
    duplicate_output_targets,
    ensure_profile_dirs,
    format_eta_duration,
    format_seconds,
    hardware_resources_from_gpus,
    load_profiles,
    load_state,
    missing_profile_dirs,
    missing_rate_fields,
    normalize_audio_codec,
    normalize_audio_container_for_codec,
    normalize_container_extension,
    normalize_profile_gpu,
    nvenc_engine_hint_from_name,
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
    variant_input_dir,
    variant_output_dir,
    variant_resource_ids,
    variant_segment_minutes,
    write_concat_file,
)
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


def test_format_eta_duration_uses_japanese_units_and_handles_edge_cases():
    assert format_eta_duration(45) == "45秒"
    assert format_eta_duration(330) == "5分30秒"
    assert format_eta_duration(3700) == "1時間1分"
    assert format_eta_duration(0) == "0秒"
    assert format_eta_duration(-5) == ""
    assert format_eta_duration(float("inf")) == ""


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

    assert "max_parallel_jobs" not in data
    assert "name" not in data
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


def test_output_and_archive_paths_expand_filename_token_with_safe_folder_names(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.output_dir = str(tmp_path / "Encoded" / "${filename}")
    profile.archive_dir = str(tmp_path / "Archive" / "${filename}")
    variant = profile.outputs[0]
    variant.folder_name = "${filename}-outputs"
    variant.filename_template = "${filename}-{height}"
    src = Path(profile.input_dir) / "Clip 01.mkv"

    output = output_path_for(profile, src, variant)

    assert output == (tmp_path / "Encoded" / "Clip-01" / "Clip-01-outputs" / "Clip 01-2160.mp4").resolve()
    assert profile_archive_dir(profile, src) == (tmp_path / "Archive" / "Clip-01").resolve()


def test_audio_codec_and_container_choices_are_normalized():
    assert audio_containers_for_codec("aac") == ["m4a", "mp4", "mov"]
    assert normalize_audio_codec("unknown") == "copy"
    assert normalize_audio_container_for_codec("aac", "mp4") == "mp4"
    assert normalize_audio_container_for_codec("aac", "mp3") == "m4a"
    assert normalize_audio_container_for_codec("opus", "webm") == "webm"

    variant = OutputVariant(
        id="audio",
        name="Audio",
        folder_name="audio",
        audio_codec="unknown",
        audio_container="mp3",
    )

    assert variant.audio_codec == "copy"
    assert variant.audio_container == "mka"


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
    assert not hasattr(profile, "name")
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

    direct = EncodeProfile(id="direct", input_dir="in", output_dir="out", archive_dir="arch", outputs=None)
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


def test_hardware_resources_only_include_detected_devices_by_default():
    resources = hardware_resources_from_gpus([])

    assert CPU_RESOURCE_ID in [resource.id for resource in resources]
    assert all(resource.backend == BACKEND_CPU for resource in resources)


def test_default_outputs_match_archive_profile(tmp_path: Path):
    profile = core_module.default_profile(
        build_paths(tmp_path),
        [GpuInfo(index=0, name="NVIDIA GeForce RTX 5080")],
    )

    assert profile.cq_value == 15
    assert Path(profile.output_dir) == tmp_path / "output" / "{source}"
    assert Path(profile.archive_dir) == tmp_path / "output" / "{source}"
    names = [(output.id, output.name, output.folder_name, output.height) for output in profile.outputs]
    assert names == [
        ("master_2160p", "UP Convert 4K", "up-convert-4k", 2160),
        ("reference_1080p", "ReEncode Original Pixel", "reencode-original-pixel", None),
    ]


def test_default_profile_outputs_follow_detected_non_nvenc_backend(tmp_path: Path):
    profile = core_module.default_profile(
        build_paths(tmp_path),
        [GpuInfo(index=0, name="Intel Arc B580", vendor="intel")],
    )

    assert profile.use_gpu is False
    assert profile.resource_ids == ["intel:0"]
    assert all(output.backend == BACKEND_QSV for output in profile.outputs)
    assert all(output.ffmpeg_encoder == "hevc_qsv" for output in profile.outputs)
    assert all(output.resource_ids == ["intel:0"] for output in profile.outputs)


def test_nvenc_engine_hint_covers_rtx_5080_and_5090():
    assert nvenc_engine_hint_from_name("NVIDIA GeForce RTX 5080") == 2
    assert nvenc_engine_hint_from_name("NVIDIA GeForce RTX 5090 Laptop GPU") == 3
    assert nvenc_engine_hint_from_name("NVIDIA GeForce RTX 5070") is None


def test_hardware_resources_use_nvenc_model_hint_when_caps_helper_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        core_module,
        "detect_nvenc_engine_count",
        lambda _index: (None, "nvEncodeAPI64.dll loaded; helper unavailable."),
    )

    resources = hardware_resources_from_gpus([GpuInfo(index=0, name="NVIDIA GeForce RTX 5080")])
    nvenc = next(resource for resource in resources if resource.id == "nvidia:0")

    assert nvenc.concurrency_slots == 2
    assert nvenc.detected_encoder_engines == 2
    assert "Model hint" in nvenc.detection_error


def test_wmi_gpu_data_adds_intel_and_amd_resources_without_duplicating_nvidia():
    existing = [GpuInfo(index=0, name="NVIDIA GeForce RTX 5080")]
    gpus = core_module.gpus_from_wmi_data(
        [
            {"Name": "NVIDIA GeForce RTX 5080", "PNPDeviceID": "PCI\\VEN_10DE", "AdapterCompatibility": "NVIDIA"},
            {"Name": "Intel Arc B580", "PNPDeviceID": "PCI\\VEN_8086", "AdapterCompatibility": "Intel"},
            {
                "Name": "AMD Radeon RX 9070",
                "PNPDeviceID": "PCI\\VEN_1002",
                "AdapterCompatibility": "Advanced Micro Devices",
            },
        ],
        existing=existing,
    )

    assert [(gpu.vendor, gpu.index, gpu.name) for gpu in gpus] == [
        ("intel", 0, "Intel Arc B580"),
        ("amd", 0, "AMD Radeon RX 9070"),
    ]


def test_cpu_resources_from_wmi_data_include_model_and_socket():
    resources = cpu_resources_from_wmi_data(
        [
            {"DeviceID": "CPU0", "Name": "Intel Xeon A", "SocketDesignation": "Socket 0"},
            {"DeviceID": "CPU1", "Name": "AMD EPYC B", "SocketDesignation": "Socket 1"},
        ]
    )

    assert [resource.id for resource in resources] == ["cpu:0", "cpu:1"]
    assert "Intel Xeon A" in resources[0].label
    assert "Socket 1" in resources[1].label


def test_detect_cpu_resources_is_cached_and_returns_copies(monkeypatch):
    clear_cpu_resource_cache()
    monkeypatch.setattr(core_module.os, "name", "nt")
    calls = []

    class Result:
        returncode = 0
        stdout = json.dumps({"DeviceID": "CPU0", "Name": "Intel Xeon A", "SocketDesignation": "Socket 0"})

    def fake_run(*_args, **_kwargs):
        calls.append(_args)
        return Result()

    monkeypatch.setattr(core_module.subprocess, "run", fake_run)
    try:
        first = detect_cpu_resources()
        injected = HardwareResource(id="cpu:99", label="Injected", kind="cpu", backend=BACKEND_CPU)
        first.append(injected)
        second = detect_cpu_resources()
    finally:
        clear_cpu_resource_cache()

    assert len(calls) == 1
    assert injected not in second
    assert [resource.id for resource in second] == ["cpu:0"]
    assert first is not second


def test_normalize_hardware_resources_keeps_explicit_manual_resource(tmp_path: Path):
    profile = EncodeProfile.from_dict(
        {
            "hardware_resources": [
                {
                    "id": "intel:0",
                    "label": "Intel GPU 0 (manual)",
                    "kind": "gpu",
                    "backend": BACKEND_QSV,
                    "vendor": "intel",
                    "detection_error": "Manual fallback resource; availability is checked before encoding.",
                }
            ],
            "resource_ids": ["intel:0"],
        },
        build_paths(tmp_path),
        [],
    )

    assert [resource.id for resource in profile.hardware_resources] == [CPU_RESOURCE_ID, "intel:0"]


def test_normalize_hardware_resources_drops_stale_detected_nvidia(tmp_path: Path):
    profile = EncodeProfile.from_dict(
        {
            "hardware_resources": [
                {
                    "id": "nvidia:0",
                    "label": "Old RTX",
                    "kind": "gpu",
                    "backend": BACKEND_NVENC,
                    "vendor": "nvidia",
                }
            ],
            "resource_ids": ["nvidia:0"],
        },
        build_paths(tmp_path),
        [],
    )

    assert CPU_RESOURCE_ID in [resource.id for resource in profile.hardware_resources]
    assert "nvidia:0" not in [resource.id for resource in profile.hardware_resources]
    assert profile.resource_ids == [CPU_RESOURCE_ID]


def test_load_profiles_defaults_missing_fields_from_supplied_paths(tmp_path: Path):
    paths = build_paths(tmp_path)
    paths.config_file.write_text(
        json.dumps({"profiles": [{"id": "partial", "name": "Partial"}]}),
        encoding="utf-8",
    )

    profiles = load_profiles(paths, [])

    assert len(profiles) == 1
    assert Path(profiles[0].input_dir) == paths.base_dir / "Incoming"
    assert Path(profiles[0].output_dir) == paths.base_dir / "output" / "{source}"
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


def test_output_variant_input_and_output_dirs_fallback_or_override(tmp_path: Path):
    profile = make_profile(tmp_path)
    variant = profile.outputs[0]

    assert variant_input_dir(profile, variant) == profile_input_dir(profile)
    assert variant_output_dir(profile, variant) == profile_output_dir(profile)

    variant.input_dir = str(tmp_path / "VariantIn")
    variant.output_dir = str(tmp_path / "VariantOut")

    assert variant_input_dir(profile, variant) == (tmp_path / "VariantIn").resolve()
    assert variant_output_dir(profile, variant) == (tmp_path / "VariantOut").resolve()


def test_scan_and_job_specs_use_variant_input_and_output_dirs(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.outputs[0].input_dir = str(tmp_path / "MasterIn")
    profile.outputs[0].output_dir = str(tmp_path / "MasterOut")
    profile.outputs[1].input_dir = str(tmp_path / "ReviewIn")
    profile.outputs[1].output_dir = str(tmp_path / "ReviewOut")
    master_src = Path(profile.outputs[0].input_dir) / "master.mkv"
    review_src = Path(profile.outputs[1].input_dir) / "review.mkv"
    master_src.parent.mkdir(parents=True)
    review_src.parent.mkdir(parents=True)
    master_src.write_bytes(b"master")
    review_src.write_bytes(b"review")

    files = scan_profile_files(profile)
    specs = build_job_specs(profile, [item.path for item in files])

    assert {item.path.name: set(item.outputs) for item in files} == {
        "master.mkv": {"master"},
        "review.mkv": {"review"},
    }
    assert [(Path(spec.src).name, spec.variant_id) for spec in specs] == [
        ("master.mkv", "master"),
        ("review.mkv", "review"),
    ]
    assert output_path_for(profile, master_src, profile.outputs[0]).is_relative_to((tmp_path / "MasterOut").resolve())


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
    edited_variant_segment = make_profile(tmp_path)
    edited_variant_segment.outputs[0].segment_minutes = 3

    base_dir = segment_dir_for(paths, src, profile, variant)
    other_profile_dir = segment_dir_for(paths, src, same_variant_other_profile, same_variant_other_profile.outputs[0])
    edited_settings_dir = segment_dir_for(paths, src, edited_settings, edited_settings.outputs[0])
    edited_variant_segment_dir = segment_dir_for(paths, src, edited_variant_segment, edited_variant_segment.outputs[0])
    src.write_bytes(b"second source content")
    replaced_source_dir = segment_dir_for(paths, src, profile, variant)

    assert base_dir != other_profile_dir
    assert base_dir != edited_settings_dir
    assert base_dir != edited_variant_segment_dir
    assert base_dir != replaced_source_dir


def test_variant_segment_minutes_falls_back_to_profile_default(tmp_path: Path):
    profile = make_profile(tmp_path)
    variant = profile.outputs[0]

    assert variant_segment_minutes(profile, variant) == profile.segment_minutes

    variant.segment_minutes = 4

    assert variant_segment_minutes(profile, variant) == 4


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


def test_encoder_app_profile_labels_are_derived_from_paths_and_ids(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.profiles = [
        EncodeProfile(
            id="profile_alpha",
            input_dir=str(tmp_path / "in-a"),
            output_dir=str(tmp_path / "out-a"),
            archive_dir=str(tmp_path / "archive-a"),
        ),
        EncodeProfile(
            id="profile_beta",
            input_dir=str(tmp_path / "in-b"),
            output_dir=str(tmp_path / "out-b"),
            archive_dir=str(tmp_path / "archive-b"),
        ),
        EncodeProfile(
            id="profile_gamma",
            input_dir=str(tmp_path / "in-c"),
            output_dir=str(tmp_path / "out-c"),
            archive_dir=str(tmp_path / "archive-c"),
        ),
    ]

    labels = EncoderApp._profile_labels(app)

    assert labels == ["in-a -> out-a (alpha)", "in-b -> out-b (beta)", "in-c -> out-c (gamma)"]
    assert EncoderApp.profile_index_by_id(app, "profile_beta") == 1


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
            self.visible = True

        def configure(self, **kwargs):
            self.kwargs = kwargs

        def grid(self):
            self.visible = True

        def grid_remove(self):
            self.visible = False

    app = EncoderApp.__new__(EncoderApp)
    app.output_rate_mode_var = Value("ABR")
    app.output_cq_entry = Widget()
    app.output_bitrate_entry = Widget()
    app.output_maxrate_entry = Widget()
    app.output_bufsize_entry = Widget()

    app.update_output_rate_controls()

    assert app.output_cq_entry.visible is False
    assert app.output_bitrate_entry.visible is True
    assert app.output_maxrate_entry.visible is False
    assert app.output_bufsize_entry.visible is False

    app.output_rate_mode_var = Value("VBR")
    app.update_output_rate_controls()

    assert app.output_cq_entry.visible is True
    assert app.output_bitrate_entry.visible is True
    assert app.output_maxrate_entry.visible is True
    assert app.output_bufsize_entry.visible is True


def test_select_compatible_resource_ids_falls_back_to_allowed_resource():
    assert select_compatible_resource_ids([CPU_RESOURCE_ID], ["nvidia:0"]) == [CPU_RESOURCE_ID]
    assert select_compatible_resource_ids([CPU_RESOURCE_ID, "nvidia:0"], ["nvidia:0"]) == ["nvidia:0"]
    assert select_compatible_resource_ids([], ["nvidia:0"]) == []


def test_backend_accepts_rate_mode_rejects_cq_for_qsv_and_amf():
    assert backend_accepts_rate_mode(BACKEND_CPU, "CQ") is True
    assert backend_accepts_rate_mode(BACKEND_NVENC, "CQ") is True
    assert backend_accepts_rate_mode(BACKEND_QSV, "CQ") is False
    assert backend_accepts_rate_mode(BACKEND_AMF, "CQ") is False
    assert backend_accepts_rate_mode(BACKEND_QSV, "VBR") is True
    assert rate_modes_for_backend(BACKEND_QSV) == ["VBR", "ABR", "CBR"]


def test_default_output_backend_uses_enabled_gpu_backend():
    assert default_output_backend_for_resource_ids([CPU_RESOURCE_ID]) == BACKEND_CPU
    assert default_output_backend_for_resource_ids(["intel:0"]) == BACKEND_QSV
    assert default_output_backend_for_resource_ids(["amd:0"]) == BACKEND_AMF
    assert default_output_backend_for_resource_ids([CPU_RESOURCE_ID, "intel:0"]) == BACKEND_QSV


def test_profile_uses_nvenc_resource_only_tracks_nvenc():
    assert profile_uses_nvenc_resource([CPU_RESOURCE_ID]) is False
    assert profile_uses_nvenc_resource(["intel:0"]) is False
    assert profile_uses_nvenc_resource(["amd:0"]) is False
    assert profile_uses_nvenc_resource(["nvidia:0"]) is True
    assert profile_uses_nvenc_resource(["intel:0", "nvidia:0"]) is True


def test_normalize_profile_keeps_qsv_output_when_profile_use_gpu_is_false(tmp_path: Path):
    profile = make_profile(tmp_path)
    profile.use_gpu = False
    profile.resource_ids = ["intel:0"]
    profile.hardware_resources = [
        HardwareResource(id=CPU_RESOURCE_ID, label="CPU", kind="cpu", backend=BACKEND_CPU),
        HardwareResource(id="intel:0", label="Intel QSV", kind="gpu", backend=BACKEND_QSV),
    ]
    profile.outputs = [
        OutputVariant(
            id="qsv",
            name="QSV",
            folder_name="qsv",
            backend=BACKEND_QSV,
            ffmpeg_encoder="hevc_qsv",
            resource_ids=["intel:0"],
        )
    ]

    normalized = normalize_profile_gpu(profile, [])

    assert normalized.use_gpu is False
    assert normalized.resource_ids == ["intel:0"]
    assert normalized.outputs[0].backend == BACKEND_QSV
    assert normalized.outputs[0].resource_ids == ["intel:0"]


def test_selected_profile_resource_ids_does_not_fallback_to_cpu():
    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    app = EncoderApp.__new__(EncoderApp)
    app.resource_enabled_vars = {
        CPU_RESOURCE_ID: Value(False),
        "intel:0": Value(False),
    }

    assert app.selected_profile_resource_ids() == []

    app.resource_enabled_vars["intel:0"] = Value(True)
    assert app.selected_profile_resource_ids() == ["intel:0"]


def test_update_output_encoder_controls_removes_cq_for_qsv():
    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    class Widget:
        def __init__(self):
            self.config = {}
            self.visible = True

        def configure(self, **kwargs):
            self.config.update(kwargs)

        def grid(self):
            self.visible = True

        def grid_remove(self):
            self.visible = False

    app = EncoderApp.__new__(EncoderApp)
    app.output_encoder_combo = Widget()
    app.output_rate_combo = Widget()
    app.output_split_combo = Widget()
    app.output_backend_var = Value(BACKEND_QSV)
    app.output_encoder_var = Value("hevc_qsv")
    app.output_codec_var = Value("hevc_nvenc")
    app.output_cpu_codec_var = Value("libx264")
    app.output_rate_mode_var = Value("CQ")
    app.output_split_encode_mode_var = Value("auto")
    app.output_cq_entry = Widget()
    app.output_bitrate_entry = Widget()
    app.output_maxrate_entry = Widget()
    app.output_bufsize_entry = Widget()
    app.encoder_capabilities = {}
    app._encoders_for_backend = lambda backend, resource_ids=None, verify_nvenc=False: ["hevc_qsv"]
    app.selected_output_resource_ids = lambda: ["intel:0"]
    app.current_profile = lambda: make_profile(Path("unused"))
    app.render_output_resource_controls = lambda *_args: None

    app.update_output_encoder_controls()

    assert app.output_rate_combo.config["values"] == ["VBR", "ABR", "CBR"]
    assert app.output_rate_mode_var.get() == "VBR"
    assert app.output_cq_entry.visible is False
    assert app.output_bitrate_entry.visible is True
    assert app.output_maxrate_entry.visible is True
    assert app.output_bufsize_entry.visible is True


def test_encoder_app_has_no_unused_ffmpeg_download_button_handler():
    assert not hasattr(EncoderApp, "download_ffmpeg_button")
    assert not hasattr(EncoderApp, "_download_ffmpeg_worker")


def test_encoder_app_has_no_unused_backend_combo_widget():
    assert "output_backend_combo" not in inspect.getsource(EncoderApp._build_profile_tab)


def test_encoder_app_has_no_unused_output_resource_change_handler():
    assert not hasattr(EncoderApp, "on_output_resource_changed")


def test_encoder_app_has_no_unused_ensure_ffmpeg_before_run_handler():
    assert not hasattr(EncoderApp, "ensure_ffmpeg_before_run")


def test_prepare_ffmpeg_and_start_rejects_when_busy(monkeypatch):
    app = EncoderApp.__new__(EncoderApp)
    app.lock = threading.Lock()
    app.running = True
    app.preparing = False
    app._sync_runtime_controls = lambda: None
    warnings = []
    threads = []

    monkeypatch.setattr(app_module.messagebox, "showwarning", lambda title, message: warnings.append((title, message)))

    class FakeThread:
        def __init__(self, *args, **kwargs):
            threads.append((args, kwargs))

        def start(self):
            raise AssertionError("worker thread should not start when busy")

    monkeypatch.setattr(app_module.threading, "Thread", FakeThread)

    app.prepare_ffmpeg_and_start(None, [])

    assert app.preparing is False
    assert threads == []
    assert len(warnings) == 1


def test_finish_ffmpeg_prepare_success_starts_when_valid(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.lock = threading.Lock()
    app.preparing = True
    app.log = lambda _text: None
    app.update_output_encoder_controls = lambda: None
    app._sync_runtime_controls = lambda: None
    app.validate_encoder_capabilities_before_run = lambda profile, specs: True
    started = []
    app.start_specs = lambda profile, specs, save: started.append((profile, specs, save))

    capabilities = {"hevc_nvenc": {"available": True}}
    profile = make_profile(tmp_path)
    specs = ["spec"]

    app._finish_ffmpeg_prepare_success(profile, specs, capabilities)

    assert app.preparing is False
    assert app.encoder_capabilities == capabilities
    assert app.encoder_smoke_cache == {}
    assert started == [(profile, specs, True)]


def test_finish_ffmpeg_prepare_success_skips_start_when_invalid(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.lock = threading.Lock()
    app.preparing = True
    app.log = lambda _text: None
    app.update_output_encoder_controls = lambda: None
    app._sync_runtime_controls = lambda: None
    app.validate_encoder_capabilities_before_run = lambda profile, specs: False
    started = []
    app.start_specs = lambda *args, **kwargs: started.append(args)

    app._finish_ffmpeg_prepare_success(make_profile(tmp_path), ["spec"], {"hevc_nvenc": {}})

    assert app.preparing is False
    assert started == []


def test_finish_ffmpeg_prepare_error_resets_preparing(monkeypatch):
    app = EncoderApp.__new__(EncoderApp)
    app.lock = threading.Lock()
    app.preparing = True
    app._sync_runtime_controls = lambda: None
    errors = []

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    app._finish_ffmpeg_prepare_error(RuntimeError("boom"))

    assert app.preparing is False
    assert any("boom" in message for _title, message in errors)


def test_runtime_controls_use_two_button_state_cycle():
    class Button:
        def __init__(self):
            self.config = {}

        def configure(self, **kwargs):
            self.config.update(kwargs)

        def grid(self):
            pass

        def grid_remove(self):
            pass

    app = EncoderApp.__new__(EncoderApp)
    app.running = False
    app.preparing = False
    app.paused = False
    app.has_saved_state = lambda: False
    app.start_button = Button()
    app.stop_button = Button()

    app._sync_runtime_controls()

    assert app.start_button.config["text"] == "開始"
    assert app.start_button.config["state"] == app_module.tk.NORMAL
    assert app.stop_button.config["state"] == app_module.tk.DISABLED

    app.running = True
    app._sync_runtime_controls()

    assert app.start_button.config["text"] == "一時停止"
    assert app.start_button.config["state"] == app_module.tk.NORMAL
    assert app.stop_button.config["state"] == app_module.tk.NORMAL

    app.paused = True
    app._sync_runtime_controls()

    assert app.start_button.config["text"] == "再開"


def test_output_editor_dialog_reuses_matching_existing_session():
    source = inspect.getsource(EncoderApp.open_output_editor_dialog)

    assert "OutputEditorSession" in source
    assert "session.show()" in source
    assert "session.focus()" in source
    assert "variant_id" in source


def test_output_editor_session_warns_on_unsaved_changes_and_persists_profiles():
    source = inspect.getsource(app_module.OutputEditorSession)

    assert "messagebox.askyesnocancel" in source
    assert "save_profiles(self.app.paths, self.app.profiles)" in source
    assert "output_editor_sessions" in source


def test_scale_flags_selector_unregisters_variable_trace_on_destroy():
    class Variable:
        def __init__(self):
            self.removed = []

        def trace_remove(self, mode, trace_id):
            self.removed.append((mode, trace_id))

    variable = Variable()
    selector = ScaleFlagsSelector.__new__(ScaleFlagsSelector)
    selector.variable = variable
    selector._trace_id = "trace-id"
    selector._destroyed = False
    event = type("Event", (), {"widget": selector})()

    selector._on_destroy(event)

    assert selector._destroyed is True
    assert selector._trace_id is None
    assert variable.removed == [("write", "trace-id")]


def test_update_output_encoder_controls_has_no_unused_cpu_codec_combo():
    assert "output_cpu_codec_combo" not in inspect.getsource(EncoderApp.update_output_encoder_controls)


def test_output_resource_dialog_has_single_instance_guard():
    source = inspect.getsource(EncoderApp.open_output_resource_dialog)

    assert "output_resource_window" in source
    assert "self._widget_exists(existing)" in source
    assert "dialog.protocol" in source


def test_surface_frame_style_keeps_inner_frames_flat():
    source = inspect.getsource(EncoderApp._configure_style)
    surface_block = source.split('"Surface.TFrame"', 1)[1].split('style.configure("Inset.TFrame"', 1)[0]

    assert 'relief="flat"' in surface_block
    assert "borderwidth=0" in surface_block
    assert 'relief="raised"' not in surface_block
    assert "borderwidth=1" not in surface_block


def test_neumorphic_palette_only_defines_used_color_tokens():
    source = inspect.getsource(EncoderApp._configure_style)
    palette_block = source.split("self.colors = {", 1)[1].split("}", 1)[0]

    assert '"sunken"' not in palette_block
    assert '"shadow"' not in palette_block


def test_set_output_edit_defaults_uses_qsv_backend_and_encoder(tmp_path: Path):
    class Value:
        def __init__(self, value=None):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            self.value = value

    app = EncoderApp.__new__(EncoderApp)
    for name in (
        "output_name_var",
        "output_folder_var",
        "output_resolution_var",
        "output_custom_height_var",
        "output_container_var",
        "output_enabled_var",
        "output_filename_template_var",
        "output_input_dir_var",
        "output_output_dir_var",
        "output_segment_minutes_var",
        "output_backend_var",
        "output_encoder_var",
        "output_split_encode_mode_var",
        "output_gpu_choice_var",
        "output_codec_var",
        "output_cpu_codec_var",
        "output_preset_var",
        "output_cpu_preset_var",
        "output_cpu_tune_var",
        "output_tune_var",
        "output_rate_mode_var",
        "output_cq_var",
        "output_bitrate_var",
        "output_maxrate_var",
        "output_bufsize_var",
        "output_pix_fmt_var",
        "output_scale_flags_var",
        "output_audio_codec_var",
        "output_audio_bitrate_var",
        "output_audio_container_var",
        "output_extra_input_args_var",
        "output_extra_video_args_var",
        "output_extra_audio_args_var",
        "output_extra_output_args_var",
        "output_extra_concat_args_var",
        "output_extra_mux_args_var",
    ):
        setattr(app, name, Value())
    app.gpus = []
    app.update_output_cpu_tune_choices = lambda: None
    app.update_output_rate_controls = lambda: None
    app.update_resolution_controls = lambda: None
    app.render_output_resource_controls = lambda *_args: None
    app.update_output_encoder_controls = lambda: None

    profile = make_profile(tmp_path)
    profile.resource_ids = ["intel:0"]

    app.set_output_edit_defaults(profile)

    assert app.output_backend_var.get() == BACKEND_QSV
    assert app.output_encoder_var.get() == "hevc_qsv"


def test_refresh_outputs_tree_inserts_full_output_tuple_once():
    class FakeTree:
        def __init__(self):
            self.deleted = None
            self.insert_calls = []

        def get_children(self):
            return ("old",)

        def delete(self, *items):
            self.deleted = items

        def insert(self, parent, index, iid=None, values=(), tags=()):
            self.insert_calls.append((parent, index, iid, values, tags))

        def item(self, *_args, **_kwargs):
            raise AssertionError("refresh_outputs_tree should not rewrite inserted values")

    app = EncoderApp.__new__(EncoderApp)
    app.outputs_tree = FakeTree()
    app.editing_outputs = [
        OutputVariant(
            id="qsv",
            name="QSV",
            folder_name="qsv",
            height=1080,
            container="mp4",
            backend=BACKEND_QSV,
            ffmpeg_encoder="hevc_qsv",
        )
    ]

    app.refresh_outputs_tree()

    assert app.outputs_tree.deleted == ("old",)
    assert len(app.outputs_tree.insert_calls) == 1
    _parent, _index, iid, values, tags = app.outputs_tree.insert_calls[0]
    assert iid == "qsv"
    assert tags == (BACKEND_QSV,)
    assert values[1:] == ("QSV", BACKEND_QSV, "hevc_qsv", "1080p", "qsv", "mp4")
    assert len(values) == 7


def test_output_bulk_enable_disable_duplicate_and_remove(tmp_path: Path):
    class FakeTree:
        def __init__(self, selected):
            self.selected = tuple(selected)
            self.selection_updates = []

        def selection(self):
            return self.selected

        def selection_set(self, *items):
            self.selection_updates.append(tuple(items))
            self.selected = tuple(items)

    app = EncoderApp.__new__(EncoderApp)
    first = OutputVariant(id="first", name="First", folder_name="first", enabled=True)
    second = OutputVariant(id="second", name="Second", folder_name="second", enabled=True)
    app.editing_outputs = [first, second]
    app.outputs_tree = FakeTree(["first", "second"])
    app.selected_output_id = None
    app.refresh_outputs_tree = lambda: None
    app.on_output_select = lambda: None

    app.set_selected_outputs_enabled(False)

    assert first.enabled is False
    assert second.enabled is False

    app.duplicate_selected_outputs()

    assert len(app.editing_outputs) == 4
    copied = app.editing_outputs[2:]
    assert [item.name for item in copied] == ["First copy", "Second copy"]
    assert copied[0].folder_name != first.folder_name
    assert copied[1].folder_name != second.folder_name

    app.outputs_tree.selected = ("first", copied[0].id)
    app.remove_output()

    assert [item.id for item in app.editing_outputs] == ["second", copied[1].id]


def test_selected_resource_labels_only_reports_selected_resources(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    profile = make_profile(tmp_path)
    profile.hardware_resources = [
        HardwareResource(id=CPU_RESOURCE_ID, label="CPU", kind="cpu", backend=BACKEND_CPU),
        HardwareResource(id="nvidia:0", label="GPU 0: RTX Test", kind="gpu", backend=BACKEND_NVENC),
    ]

    assert app.selected_resource_labels(profile, [CPU_RESOURCE_ID]) == ["CPU"]
    assert app.selected_resource_labels(profile, ["nvidia:0"]) == ["GPU 0: RTX Test"]


def test_add_or_update_output_rejects_cq_for_qsv(tmp_path: Path, monkeypatch):
    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    app = EncoderApp.__new__(EncoderApp)
    profile = make_profile(tmp_path)
    app.current_profile = lambda: profile
    app.selected_output_id = None
    app.editing_outputs = []
    app.output_name_var = Value("QSV")
    app.output_folder_var = Value("qsv")
    app.output_container_var = Value("mp4")
    app.output_resolution_var = Value("Original")
    app.output_segment_minutes_var = Value("")
    app.output_rate_mode_var = Value("CQ")
    app.output_cq_var = Value("22")
    app.cq_var = Value("22")
    app.output_backend_var = Value(BACKEND_QSV)
    app.output_encoder_var = Value("hevc_qsv")
    errors = []

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    app.add_or_update_output()

    assert app.editing_outputs == []
    assert any("CQ" in message and "QSV/AMF" in message for _title, message in errors)


def test_add_or_update_output_allows_qsv_vbr_with_blank_cq(tmp_path: Path, monkeypatch):
    class Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    class Tree:
        def __init__(self):
            self.selection = None

        def selection_set(self, item):
            self.selection = item

    app = EncoderApp.__new__(EncoderApp)
    profile = make_profile(tmp_path)
    app.current_profile = lambda: profile
    app.selected_output_id = None
    app.editing_outputs = []
    app.output_name_var = Value("QSV")
    app.output_folder_var = Value("qsv")
    app.output_container_var = Value("mp4")
    app.output_resolution_var = Value("Original")
    app.output_rate_mode_var = Value("VBR")
    app.output_cq_var = Value("")
    app.cq_var = Value("")
    app.output_backend_var = Value(BACKEND_QSV)
    app.output_encoder_var = Value("hevc_qsv")
    app.selected_output_resource_ids = lambda: ["intel:0"]
    app.output_enabled_var = Value(True)
    app.output_filename_template_var = Value("{source}")
    app.output_input_dir_var = Value("")
    app.output_output_dir_var = Value("")
    app.output_segment_minutes_var = Value("")
    app.output_codec_var = Value("hevc_nvenc")
    app.output_cpu_codec_var = Value("libx264")
    app.output_split_encode_mode_var = Value("auto")
    app.output_preset_var = Value("p5")
    app.output_cpu_preset_var = Value("medium")
    app.output_cpu_tune_var = Value("none")
    app.output_tune_var = Value("none")
    app.output_bitrate_var = Value("6000k")
    app.output_maxrate_var = Value("8000k")
    app.output_bufsize_var = Value("12000k")
    app.output_pix_fmt_var = Value("yuv420p")
    app.output_scale_flags_var = Value("lanczos")
    app.output_audio_codec_var = Value("copy")
    app.output_audio_bitrate_var = Value("")
    app.output_audio_container_var = Value("")
    app.output_extra_input_args_var = Value("")
    app.output_extra_video_args_var = Value("")
    app.output_extra_audio_args_var = Value("")
    app.output_extra_output_args_var = Value("")
    app.output_extra_concat_args_var = Value("")
    app.output_extra_mux_args_var = Value("")
    app.refresh_outputs_tree = lambda: None
    app.outputs_tree = Tree()
    errors = []

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    app.add_or_update_output()

    assert errors == []
    assert len(app.editing_outputs) == 1
    variant = app.editing_outputs[0]
    assert variant.backend == BACKEND_QSV
    assert variant.rate_mode == "VBR"
    assert variant.cq_value == profile.cq_value
    assert variant.scale_flags == ""
    assert app.outputs_tree.selection == variant.id


def test_validate_profile_before_run_rejects_cq_for_qsv(tmp_path: Path, monkeypatch):
    app = EncoderApp.__new__(EncoderApp)
    profile = make_profile(tmp_path)
    profile.hardware_resources = [
        HardwareResource(id=CPU_RESOURCE_ID, label="CPU", kind="cpu", backend=BACKEND_CPU),
        HardwareResource(id="intel:0", label="Intel QSV", kind="gpu", backend=BACKEND_QSV),
    ]
    profile.resource_ids = ["intel:0"]
    profile.outputs[0].backend = BACKEND_QSV
    profile.outputs[0].ffmpeg_encoder = "hevc_qsv"
    profile.outputs[0].resource_ids = ["intel:0"]
    profile.outputs[0].rate_mode = "CQ"
    profile.outputs[1].enabled = False
    errors = []

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    assert app.validate_profile_before_run(profile) is False
    assert any("CQ" in message and "QSV/AMF" in message for _title, message in errors)


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
    spec = JobSpec(
        src=str(tmp_path / "input.mkv"), profile_id=profile.id, variant_id=variant.id, assigned_resource_id="nvidia:0"
    )

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
    spec = JobSpec(
        src=str(tmp_path / "input.mkv"), profile_id=profile.id, variant_id=variant.id, assigned_resource_id="nvidia:0"
    )

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))
    monkeypatch.setattr(
        app_module, "smoke_test_encoder", lambda *_args, **_kwargs: smoke_calls.append(_args) or (True, "")
    )

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
    spec = JobSpec(
        src=str(tmp_path / "input.mkv"), profile_id=profile.id, variant_id=variant.id, assigned_resource_id="nvidia:0"
    )

    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, message: errors.append((title, message)))

    def fake_smoke(ffmpeg_path, encoder, resource_id="", split_encode_mode="", timeout=30):
        smoke_calls.append((encoder, resource_id, split_encode_mode))
        return True, ""

    monkeypatch.setattr(app_module, "smoke_test_encoder", fake_smoke)

    assert app.split_encode_modes_for_encoder("hevc_nvenc") == ["auto", "disabled"]
    assert app.validate_encoder_capabilities_before_run(profile, [spec]) is True
    assert errors == []
    assert smoke_calls == [("hevc_nvenc", "nvidia:0", "disabled")]


def test_nvenc_encoder_choices_use_common_resource_support(tmp_path: Path, monkeypatch):
    app = EncoderApp.__new__(EncoderApp)
    app.paths = build_paths(tmp_path)
    app.encoder_smoke_cache = {}
    app.encoder_capabilities = {
        "h264_nvenc": {"available": True, "supports_split_encode_mode": False, "split_encode_modes": []},
        "hevc_nvenc": {"available": True, "supports_split_encode_mode": True, "split_encode_modes": ["auto"]},
        "av1_nvenc": {"available": True, "supports_split_encode_mode": True, "split_encode_modes": ["auto"]},
    }

    def fake_smoke(_ffmpeg_path, encoder, resource_id="", split_encode_mode="", timeout=30):
        if encoder == "av1_nvenc" and resource_id == "nvidia:0":
            return False, "unsupported"
        return True, ""

    monkeypatch.setattr(app_module, "smoke_test_encoder", fake_smoke)

    encoders = app._encoders_for_backend(BACKEND_NVENC, ["nvidia:0", "nvidia:1"], verify_nvenc=True)

    assert encoders == ["h264_nvenc", "hevc_nvenc"]


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


def test_resource_detail_label_only_reports_reserved_slot_indexes(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
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

    assert app.resource_detail_label(job) == "nvidia:0"

    job.resource_slot_indexes = [0]
    assert app.resource_detail_label(job) == "nvidia:0 slot 1"

    job.resource_slot_indexes = [0, 2]
    assert app.resource_detail_label(job) == "nvidia:0 slots 1,3"


def test_resource_slot_indices_do_not_overlap_after_release(tmp_path: Path):
    app = EncoderApp.__new__(EncoderApp)
    app.lock = threading.Lock()
    app.active_resource_slots = {}
    app.active_resource_slot_indexes = {}
    app.active_jobs = {}
    profile = make_profile(tmp_path)

    def make_job(job_id: int) -> RuntimeJob:
        return RuntimeJob(
            job_id=job_id,
            spec=JobSpec(
                src=str(tmp_path / f"input-{job_id}.mkv"), profile_id=profile.id, variant_id=profile.outputs[0].id
            ),
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
    assert app.reserve_resource_slots(second, {"nvidia:0": 3}) is False
    assert first.resource_slot_indexes == [0, 1, 2]

    app.release_resource_slots(first)
    assert app.active_resource_slot_indexes == {}

    assert app.reserve_resource_slots(third, {"nvidia:0": 3}) is True
    assert third.resource_slot_indexes == [0, 1, 2]


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


def test_scheduler_stop_discard_clears_state_and_temporary_outputs(tmp_path: Path):
    paths = build_paths(tmp_path)
    profile = make_profile(tmp_path)
    variant = profile.outputs[0]
    src = Path(profile.input_dir) / "video.mkv"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"dummy")
    spec = JobSpec(src=str(src), profile_id=profile.id, variant_id=variant.id)
    save_state(paths, profile, [spec])

    app = EncoderApp.__new__(EncoderApp)
    app.lock = threading.Lock()
    app.paths = paths
    app.running = True
    app.paused = True
    app.stop_requested = True
    app.discard_state_on_stop = True
    app.active_jobs = {}
    app.pending_jobs = app_module.queue.Queue()
    app.log_messages = []
    app.log = app.log_messages.append
    app.scan_files_called = False
    app.scan_files = lambda: setattr(app, "scan_files_called", True)

    class Root:
        def after(self, _delay, callback=None):
            if callback is not None:
                callback()

    app.root = Root()
    job = RuntimeJob(
        job_id=1,
        spec=spec,
        profile=profile,
        variant=variant,
        tmp_out=tmp_path / "tmp.mp4",
        out_file=tmp_path / "out.mp4",
        log_file=tmp_path / "job.log",
    )
    app.all_jobs = {job.job_id: job}

    temporary_paths = [
        core_module.joined_video_path_for(paths, src, profile, variant),
        core_module.temp_audio_path_for(paths, src, profile, variant),
        core_module.temp_output_path_for(paths, src, profile, variant),
    ]
    for path in temporary_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"temporary")
    segment_dir = segment_dir_for(paths, src, profile, variant)
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "segment-00001.mp4").write_bytes(b"segment")
    unrelated_segment = paths.tmp_dir / "segments" / "unrelated" / "segment-00001.mp4"
    unrelated_segment.parent.mkdir(parents=True, exist_ok=True)
    unrelated_segment.write_bytes(b"keep")

    app.scheduler_loop(profile)

    assert not paths.state_file.exists()
    assert all(not path.exists() for path in temporary_paths)
    assert not segment_dir.exists()
    assert unrelated_segment.exists()
    assert app.running is False
    assert app.paused is False
    assert app.discard_state_on_stop is False
    assert app.scan_files_called is True
    assert any("破棄" in message for message in app.log_messages)


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
     future-mode     4            E..V.......
  -gpu <int>         E..V....... Selects which NVENC capable GPU to use
"""

    assert split_encode_modes_from_help(help_text) == ["auto", "disabled", "forced", "2", "future-mode"]
    assert split_encode_modes_from_help("h264 mentions 2 but no option") == []


def test_smoke_test_encoder_uses_nvenc_safe_frame_size(monkeypatch, tmp_path: Path):
    calls = []

    class Result:
        returncode = 0
        stdout = ""

    def fake_run(command, **_kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    ok, detail = downloader.smoke_test_encoder(tmp_path / "ffmpeg.exe", "hevc_nvenc", resource_id="nvidia:0")

    assert ok is True
    assert detail == ""
    command = calls[0]
    assert "color=size=320x240:rate=30:duration=1" in command
    assert "color=size=64x64:rate=1:duration=1" not in command
    assert command[command.index("-gpu") + 1] == "0"


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
