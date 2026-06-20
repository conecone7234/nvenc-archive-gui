from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .subprocess_utils import no_window_subprocess_kwargs

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
SEGMENT_SOURCE_STEM_LIMIT = 40
BACKEND_CPU = "cpu"
BACKEND_NVENC = "nvenc"
BACKEND_QSV = "qsv"
BACKEND_AMF = "amf"
BACKENDS = {BACKEND_CPU, BACKEND_NVENC, BACKEND_QSV, BACKEND_AMF}
CPU_RESOURCE_ID = "cpu:0"
DEFAULT_ENCODER_BY_BACKEND = {
    BACKEND_CPU: "libx264",
    BACKEND_NVENC: "hevc_nvenc",
    BACKEND_QSV: "hevc_qsv",
    BACKEND_AMF: "hevc_amf",
}
NVIDIA_NVENC_ENGINE_HINTS = [
    (re.compile(r"\brtx\s+5090\b", re.IGNORECASE), 3),
    (re.compile(r"\brtx\s+5080\b", re.IGNORECASE), 2),
    (re.compile(r"\brtx\s+5070\s*ti\b", re.IGNORECASE), 2),
    (re.compile(r"\brtx\s+4090\b", re.IGNORECASE), 2),
    (re.compile(r"\brtx\s+4080\b", re.IGNORECASE), 2),
    (re.compile(r"\brtx\s+4070\s*ti\b", re.IGNORECASE), 2),
]
AUDIO_CODEC_CHOICES = ["copy", "aac", "alac", "libmp3lame", "mp3", "opus", "vorbis", "flac"]
STREAM_KINDS = {"video", "audio", "subtitle", "attachment", "data"}
STREAM_ACTIONS = {"transcode", "copy", "auto", "exclude"}
VOLATILE_STREAM_METADATA_TAGS = {"encoder", "duration", "number_of_bytes", "number_of_frames"}
STREAM_SCOPED_EXTRA_OPTIONS = {
    "-ar",
    "-ac",
    "-af",
    "-aq",
    "-b",
    "-bsf",
    "-channel_layout",
    "-compression_level",
    "-filter",
    "-frames",
    "-metadata",
    "-profile",
    "-q",
    "-sample_fmt",
}
AUDIO_CONTAINERS_BY_CODEC = {
    "copy": ["mka", "mkv"],
    "aac": ["m4a", "mp4", "mov"],
    "alac": ["m4a", "mp4", "mov"],
    "libmp3lame": ["mp3"],
    "mp3": ["mp3"],
    "opus": ["mka", "mkv", "webm"],
    "vorbis": ["mka", "mkv", "webm"],
    "flac": ["mka", "mkv"],
}
SOURCE_TEMPLATE_TOKENS = ("${filename}", "{source}")
STALE_AUTO_RESOURCE_ERRORS = {
    "Manual resource; availability is checked before encoding.",
}
MANUAL_RESOURCE_ERROR_PREFIX = "Manual fallback resource;"
DEPRECATED_PROFILE_FIELDS = {
    "max_parallel_jobs",
    "use_gpu",
    "gpu_index",
    "gpu_name",
    "codec",
    "cpu_codec",
    "preset",
    "cpu_preset",
    "tune",
    "cpu_tune",
    "rate_mode",
    "cq_value",
    "bitrate",
    "maxrate",
    "bufsize",
    "pix_fmt",
    "scale_flags",
}


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
    vendor: str = "nvidia"
    encoder_engines: Optional[int] = None
    detection_error: str = ""

    def __post_init__(self) -> None:
        self.index = normalize_int(self.index, minimum=0, default=0)
        self.name = str(self.name or "").strip()
        self.vendor = str(self.vendor or "nvidia").strip().lower()
        self.encoder_engines = normalize_optional_int(self.encoder_engines, minimum=1)
        self.detection_error = str(self.detection_error or "").strip()


@dataclass
class HardwareResource:
    id: str
    label: str
    kind: str
    backend: str
    vendor: str = ""
    index: int = 0
    concurrency_slots: int = 1
    detected_encoder_engines: Optional[int] = None
    detection_error: str = ""

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.label = str(self.label or self.id).strip()
        self.kind = str(self.kind or "").strip().lower()
        self.backend = normalize_backend(self.backend)
        self.vendor = str(self.vendor or "").strip().lower()
        self.index = normalize_int(self.index, minimum=0, default=0)
        self.concurrency_slots = normalize_int(self.concurrency_slots, minimum=1, default=1)
        self.detected_encoder_engines = normalize_optional_int(self.detected_encoder_engines, minimum=1)
        self.detection_error = str(self.detection_error or "").strip()


@dataclass
class ChapterInfo:
    id: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_ffprobe(data: object) -> "ChapterInfo":
        item = data if isinstance(data, dict) else {}
        return ChapterInfo(
            id=normalize_int(item.get("id", 0), minimum=0, default=0),
            start_time=normalize_float(item.get("start_time", 0.0), default=0.0),
            end_time=normalize_float(item.get("end_time", 0.0), default=0.0),
            tags=normalize_tags(item.get("tags")),
        )


@dataclass
class MediaStream:
    index: int
    codec_type: str
    codec_name: str = ""
    ordinal: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    sample_rate: str = ""
    start_time: float = 0.0
    duration: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    disposition: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = normalize_int(self.index, minimum=0, default=0)
        self.codec_type = str(self.codec_type or "data").strip().lower()
        self.codec_name = str(self.codec_name or "").strip().lower()
        self.ordinal = normalize_int(self.ordinal, minimum=0, default=0)
        self.width = normalize_optional_int(self.width, minimum=1)
        self.height = normalize_optional_int(self.height, minimum=1)
        self.channels = normalize_optional_int(self.channels, minimum=1)
        self.sample_rate = str(self.sample_rate or "").strip()
        self.start_time = normalize_float(self.start_time, default=0.0)
        self.duration = normalize_optional_float(self.duration, minimum=0.0)
        self.tags = normalize_tags(self.tags)
        raw_disposition = self.disposition if isinstance(self.disposition, dict) else {}
        self.disposition = {str(key): bool(value) for key, value in raw_disposition.items() if bool(value)}

    @property
    def attached_pic(self) -> bool:
        return bool(self.disposition.get("attached_pic"))

    @property
    def language(self) -> str:
        return self.tags.get("language", "").strip().lower()

    @property
    def title(self) -> str:
        return self.tags.get("title", "").strip()

    @staticmethod
    def from_ffprobe(data: object, ordinal: int) -> "MediaStream":
        item = data if isinstance(data, dict) else {}
        return MediaStream(
            index=normalize_int(item.get("index", 0), minimum=0, default=0),
            codec_type=str(item.get("codec_type", "data")),
            codec_name=str(item.get("codec_name", "")),
            ordinal=ordinal,
            width=item.get("width"),
            height=item.get("height"),
            channels=item.get("channels"),
            sample_rate=str(item.get("sample_rate", "")),
            start_time=item.get("start_time", 0.0),
            duration=item.get("duration"),
            tags=normalize_tags(item.get("tags")),
            disposition=item.get("disposition", {}) if isinstance(item.get("disposition"), dict) else {},
        )


@dataclass
class MediaProbe:
    path: str
    format_name: str = ""
    duration: Optional[float] = None
    streams: List[MediaStream] = field(default_factory=list)
    chapters: List[ChapterInfo] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_ffprobe(path: Path, data: object) -> "MediaProbe":
        root = data if isinstance(data, dict) else {}
        raw_streams = root.get("streams", []) if isinstance(root.get("streams"), list) else []
        ordinals: Dict[str, int] = {}
        streams: List[MediaStream] = []
        for raw in raw_streams:
            codec_type = str(raw.get("codec_type", "data")) if isinstance(raw, dict) else "data"
            ordinal = ordinals.get(codec_type, 0)
            ordinals[codec_type] = ordinal + 1
            streams.append(MediaStream.from_ffprobe(raw, ordinal))
        raw_format = root.get("format", {}) if isinstance(root.get("format"), dict) else {}
        return MediaProbe(
            path=str(path),
            format_name=str(raw_format.get("format_name", "")),
            duration=normalize_optional_float(raw_format.get("duration"), minimum=0.0),
            streams=streams,
            chapters=[ChapterInfo.from_ffprobe(item) for item in root.get("chapters", []) if isinstance(item, dict)],
            tags=normalize_tags(raw_format.get("tags")),
        )


@dataclass
class StreamSelector:
    kind: str = "video"
    ordinal: Optional[int] = None
    codec_names: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    title_contains: str = ""
    dispositions: List[str] = field(default_factory=list)
    attached_pic: Optional[bool] = None

    def __post_init__(self) -> None:
        self.kind = str(self.kind or "video").strip().lower()
        if self.kind not in STREAM_KINDS:
            self.kind = "data"
        self.ordinal = normalize_optional_int(self.ordinal, minimum=0)
        self.codec_names = normalize_string_list(self.codec_names)
        self.languages = normalize_string_list(self.languages)
        self.title_contains = str(self.title_contains or "").strip().lower()
        self.dispositions = normalize_string_list(self.dispositions)
        self.attached_pic = normalize_optional_bool(self.attached_pic)

    def matches(self, stream: MediaStream) -> bool:
        if stream.codec_type != self.kind:
            return False
        if self.ordinal is not None and stream.ordinal != self.ordinal:
            return False
        if self.codec_names and stream.codec_name not in self.codec_names:
            return False
        if self.languages and stream.language not in self.languages:
            return False
        if self.title_contains and self.title_contains not in stream.title.lower():
            return False
        if self.dispositions and any(not stream.disposition.get(name, False) for name in self.dispositions):
            return False
        if self.attached_pic is not None and stream.attached_pic != self.attached_pic:
            return False
        return True

    @staticmethod
    def from_dict(data: object) -> "StreamSelector":
        return StreamSelector(**dataclass_values(StreamSelector, data))


@dataclass
class StreamEncodingOverride:
    action: str = "auto"
    backend: str = ""
    ffmpeg_encoder: str = ""
    resource_ids: List[str] = field(default_factory=list)
    height: Optional[int] = None
    rate_mode: str = ""
    cq_value: Optional[int] = None
    bitrate: str = ""
    maxrate: str = ""
    bufsize: str = ""
    preset: str = ""
    tune: str = ""
    pix_fmt: str = ""
    scale_flags: str = ""
    codec: str = ""
    extra_args: str = ""

    def __post_init__(self) -> None:
        self.action = str(self.action or "auto").strip().lower()
        if self.action not in STREAM_ACTIONS:
            self.action = "auto"
        self.backend = normalize_backend(self.backend) if self.backend else ""
        self.ffmpeg_encoder = str(self.ffmpeg_encoder or "").strip()
        self.resource_ids = normalize_resource_id_list(self.resource_ids)
        self.height = normalize_optional_int(self.height, minimum=1)
        self.rate_mode = str(self.rate_mode or "").strip().upper()
        self.cq_value = normalize_optional_int(self.cq_value)
        self.bitrate = str(self.bitrate or "").strip()
        self.maxrate = str(self.maxrate or "").strip()
        self.bufsize = str(self.bufsize or "").strip()
        self.preset = str(self.preset or "").strip()
        self.tune = str(self.tune or "").strip()
        self.pix_fmt = str(self.pix_fmt or "").strip()
        self.scale_flags = str(self.scale_flags or "").strip()
        self.codec = str(self.codec or "").strip()
        self.extra_args = str(self.extra_args or "").strip()

    @staticmethod
    def from_dict(data: object) -> "StreamEncodingOverride":
        return StreamEncodingOverride(**dataclass_values(StreamEncodingOverride, data))


@dataclass
class StreamRule:
    id: str = ""
    name: str = ""
    selector: StreamSelector = field(default_factory=StreamSelector)
    encoding: StreamEncodingOverride = field(default_factory=StreamEncodingOverride)

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip() or new_id("stream-rule")
        if not isinstance(self.selector, StreamSelector):
            self.selector = StreamSelector.from_dict(self.selector)
        if not isinstance(self.encoding, StreamEncodingOverride):
            self.encoding = StreamEncodingOverride.from_dict(self.encoding)
        self.name = str(self.name or "").strip() or self.selector.kind

    @staticmethod
    def from_dict(data: object) -> "StreamRule":
        item = data if isinstance(data, dict) else {}
        return StreamRule(
            id=str(item.get("id", "")),
            name=str(item.get("name", "")),
            selector=StreamSelector.from_dict(item.get("selector", {})),
            encoding=StreamEncodingOverride.from_dict(item.get("encoding", {})),
        )


def default_stream_rules() -> List[StreamRule]:
    return [
        StreamRule(
            name="すべての通常映像",
            selector=StreamSelector(kind="video", attached_pic=False),
            encoding=StreamEncodingOverride(action="transcode"),
        ),
        StreamRule(
            name="カバー画像",
            selector=StreamSelector(kind="video", attached_pic=True),
            encoding=StreamEncodingOverride(action="auto"),
        ),
        StreamRule(
            name="すべての音声",
            selector=StreamSelector(kind="audio"),
            encoding=StreamEncodingOverride(action="auto"),
        ),
        StreamRule(
            name="すべての字幕",
            selector=StreamSelector(kind="subtitle"),
            encoding=StreamEncodingOverride(action="auto"),
        ),
        StreamRule(
            name="すべての添付",
            selector=StreamSelector(kind="attachment"),
            encoding=StreamEncodingOverride(action="copy"),
        ),
        StreamRule(
            name="すべてのデータ",
            selector=StreamSelector(kind="data"),
            encoding=StreamEncodingOverride(action="copy"),
        ),
    ]


@dataclass
class OutputVariant:
    id: str
    name: str
    folder_name: str
    height: Optional[int] = None
    container: str = "mp4"
    enabled: bool = True
    filename_template: str = "{source}"
    input_dir: str = ""
    output_dir: str = ""
    segment_minutes: Optional[int] = None
    use_gpu: Optional[bool] = None
    gpu_index: Optional[int] = None
    gpu_name: str = ""
    codec: str = ""
    cpu_codec: str = ""
    preset: str = ""
    cpu_preset: str = ""
    cpu_tune: str = ""
    tune: str = ""
    rate_mode: str = ""
    cq_value: Optional[int] = None
    bitrate: str = ""
    maxrate: str = ""
    bufsize: str = ""
    pix_fmt: str = ""
    scale_flags: str = ""
    audio_codec: str = ""
    audio_bitrate: str = ""
    audio_container: str = ""
    backend: str = ""
    ffmpeg_encoder: str = ""
    resource_ids: List[str] = field(default_factory=list)
    concurrency_policy: str = "resource_slots"
    split_encode_mode: str = ""
    extra_input_args: str = ""
    extra_video_args: str = ""
    extra_audio_args: str = ""
    extra_output_args: str = ""
    extra_concat_args: str = ""
    extra_mux_args: str = ""
    stream_rules: List[StreamRule] = field(default_factory=default_stream_rules)
    preserve_metadata: bool = True
    preserve_chapters: bool = True

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
        self.filename_template = str(self.filename_template or "{source}").strip() or "{source}"
        self.input_dir = str(self.input_dir or "").strip()
        self.output_dir = str(self.output_dir or "").strip()
        self.segment_minutes = normalize_optional_int(self.segment_minutes, minimum=1)
        self.use_gpu = normalize_optional_bool(self.use_gpu)
        self.gpu_index = normalize_optional_int(self.gpu_index, minimum=0)
        self.gpu_name = str(self.gpu_name or "").strip()
        self.codec = str(self.codec or "").strip()
        self.cpu_codec = str(self.cpu_codec or "").strip()
        self.preset = str(self.preset or "").strip()
        self.cpu_preset = str(self.cpu_preset or "").strip()
        self.cpu_tune = str(self.cpu_tune or "").strip()
        self.tune = str(self.tune or "").strip()
        self.rate_mode = str(self.rate_mode or "").strip().upper()
        self.cq_value = normalize_optional_int(self.cq_value)
        self.bitrate = str(self.bitrate or "").strip()
        self.maxrate = str(self.maxrate or "").strip()
        self.bufsize = str(self.bufsize or "").strip()
        self.pix_fmt = str(self.pix_fmt or "").strip()
        self.scale_flags = str(self.scale_flags or "").strip()
        self.audio_codec = normalize_audio_codec(self.audio_codec)
        self.audio_bitrate = str(self.audio_bitrate or "").strip()
        self.audio_container = normalize_audio_container_for_codec(self.audio_codec, self.audio_container)
        self.ffmpeg_encoder = str(self.ffmpeg_encoder or "").strip()
        inherit_profile_device = (
            not self.backend and not self.ffmpeg_encoder and self.use_gpu is None and not self.resource_ids
        )
        if inherit_profile_device:
            self.backend = ""
        elif not self.backend:
            if self.ffmpeg_encoder:
                self.backend = backend_from_encoder(self.ffmpeg_encoder)
            elif self.use_gpu is True:
                self.backend = BACKEND_NVENC
            else:
                self.backend = BACKEND_CPU
        if self.backend:
            self.backend = normalize_backend(self.backend)
        if not self.ffmpeg_encoder:
            if not self.backend:
                self.ffmpeg_encoder = ""
            elif self.backend == BACKEND_NVENC:
                self.ffmpeg_encoder = self.codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_NVENC]
            elif self.backend == BACKEND_CPU:
                self.ffmpeg_encoder = self.cpu_codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU]
            else:
                self.ffmpeg_encoder = DEFAULT_ENCODER_BY_BACKEND[self.backend]
        if self.backend == BACKEND_NVENC and not self.codec:
            self.codec = self.ffmpeg_encoder
        if self.backend == BACKEND_CPU and not self.cpu_codec:
            self.cpu_codec = self.ffmpeg_encoder
        if not self.resource_ids and self.backend:
            if self.backend == BACKEND_CPU:
                self.resource_ids = [CPU_RESOURCE_ID]
            elif self.backend == BACKEND_NVENC:
                self.resource_ids = [resource_id_for_backend(BACKEND_NVENC, self.gpu_index or 0)]
            else:
                self.resource_ids = [resource_id_for_backend(self.backend, 0)]
        self.resource_ids = normalize_resource_id_list(self.resource_ids)
        self.concurrency_policy = str(self.concurrency_policy or "resource_slots").strip() or "resource_slots"
        self.split_encode_mode = str(self.split_encode_mode or "").strip().lower()
        self.extra_input_args = str(self.extra_input_args or "").strip()
        self.extra_video_args = str(self.extra_video_args or "").strip()
        self.extra_audio_args = str(self.extra_audio_args or "").strip()
        self.extra_output_args = str(self.extra_output_args or "").strip()
        self.extra_concat_args = str(self.extra_concat_args or "").strip()
        self.extra_mux_args = str(self.extra_mux_args or "").strip()
        raw_rules = self.stream_rules if isinstance(self.stream_rules, list) else []
        self.stream_rules = [
            item if isinstance(item, StreamRule) else StreamRule.from_dict(item)
            for item in raw_rules
            if isinstance(item, (StreamRule, dict))
        ]
        if not self.stream_rules:
            self.stream_rules = default_stream_rules()
        seen_rule_ids: set[str] = set()
        for rule in self.stream_rules:
            while rule.id in seen_rule_ids:
                rule.id = new_id("stream-rule")
            seen_rule_ids.add(rule.id)
        self.preserve_metadata = normalize_bool(self.preserve_metadata, default=True)
        self.preserve_chapters = normalize_bool(self.preserve_chapters, default=True)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "OutputVariant":
        base = {
            "id": "",
            "name": "",
            "folder_name": "",
            "height": None,
            "container": "mp4",
            "enabled": True,
            "input_dir": "",
            "output_dir": "",
            "segment_minutes": None,
        }
        base.update(dataclass_values(OutputVariant, data))
        raw_rules = base.get("stream_rules")
        base["stream_rules"] = (
            [StreamRule.from_dict(item) for item in raw_rules if isinstance(item, dict)]
            if isinstance(raw_rules, list)
            else default_stream_rules()
        )
        return OutputVariant(**base)


EncodeSet = OutputVariant


@dataclass
class EncodeProfile:
    id: str
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
    cpu_tune: str = "none"
    rate_mode: str = "CQ"
    cq_value: int = 18
    bitrate: str = "25000k"
    maxrate: str = "40000k"
    bufsize: str = "80000k"
    pix_fmt: str = "nv12"
    scale_flags: str = "lanczos+accurate_rnd"
    resource_ids: List[str] = field(default_factory=list)
    hardware_resources: List[HardwareResource] = field(default_factory=list)
    outputs: List[OutputVariant] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip() or new_id("profile")
        self.input_dir = str(self.input_dir or "").strip()
        self.output_dir = str(self.output_dir or "").strip()
        self.archive_dir = str(self.archive_dir or "").strip()
        self.gpu_name = str(self.gpu_name or "").strip()
        self.codec = str(self.codec or "hevc_nvenc").strip()
        self.cpu_codec = str(self.cpu_codec or "libx264").strip()
        self.preset = str(self.preset or "p7").strip()
        self.cpu_preset = str(self.cpu_preset or "medium").strip()
        self.tune = str(self.tune or "hq").strip()
        self.cpu_tune = str(self.cpu_tune or "none").strip()
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

        self.resource_ids = normalize_resource_id_list(self.resource_ids)
        raw_resources = self.hardware_resources if isinstance(self.hardware_resources, list) else []
        resources: List[HardwareResource] = []
        seen_resource_ids: set[str] = set()
        for resource in raw_resources:
            if isinstance(resource, HardwareResource):
                item = resource
            elif isinstance(resource, dict):
                values = {
                    "id": "",
                    "label": "",
                    "kind": "",
                    "backend": "",
                }
                values.update(dataclass_values(HardwareResource, resource))
                item = HardwareResource(**values)
            else:
                continue
            if not item.id or item.id in seen_resource_ids:
                continue
            seen_resource_ids.add(item.id)
            resources.append(item)
        self.hardware_resources = resources

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
        explicit_values = dataclass_values(EncodeProfile, data)
        base.update(explicit_values)
        raw_outputs = base.get("outputs")
        if not isinstance(raw_outputs, list):
            raw_outputs = []
        base["outputs"] = [OutputVariant.from_dict(x) for x in raw_outputs]
        if not base["outputs"]:
            base["outputs"] = default_outputs()
        profile = normalize_profile_gpu(EncodeProfile(**base), default_gpus)
        if not profile.use_gpu:
            cpu_defaults = default_profile(base_paths, [])
            for key in ("max_parallel_jobs", "cq_value", "bitrate", "maxrate", "bufsize", "cpu_preset", "cpu_tune"):
                if key not in explicit_values:
                    setattr(profile, key, getattr(cpu_defaults, key))
        return profile


@dataclass
class VideoStreamTask:
    input_stream_index: int
    input_ordinal: int
    output_ordinal: int
    source_codec: str = ""
    start_time: float = 0.0
    duration: Optional[float] = None
    expected_codec: str = ""
    expected_width: Optional[int] = None
    expected_height: Optional[int] = None
    settings: StreamEncodingOverride = field(default_factory=lambda: StreamEncodingOverride(action="transcode"))
    use_muxer_default: bool = False
    fallback_reason: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    disposition: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_stream_index = normalize_int(self.input_stream_index, minimum=0, default=0)
        self.input_ordinal = normalize_int(self.input_ordinal, minimum=0, default=0)
        self.output_ordinal = normalize_int(self.output_ordinal, minimum=0, default=0)
        self.source_codec = str(self.source_codec or "").strip().lower()
        self.start_time = normalize_float(self.start_time, default=0.0)
        self.duration = normalize_optional_float(self.duration, minimum=0.0)
        self.expected_codec = str(self.expected_codec or "").strip().lower()
        self.expected_width = normalize_optional_int(self.expected_width, minimum=1)
        self.expected_height = normalize_optional_int(self.expected_height, minimum=1)
        if not isinstance(self.settings, StreamEncodingOverride):
            self.settings = StreamEncodingOverride.from_dict(self.settings)
        self.use_muxer_default = normalize_bool(self.use_muxer_default)
        self.fallback_reason = str(self.fallback_reason or "").strip()
        self.tags = normalize_tags(self.tags)
        raw_disposition = self.disposition if isinstance(self.disposition, dict) else {}
        self.disposition = {str(key): bool(value) for key, value in raw_disposition.items() if bool(value)}

    @staticmethod
    def from_dict(data: object) -> "VideoStreamTask":
        item = data if isinstance(data, dict) else {}
        return VideoStreamTask(
            input_stream_index=item.get("input_stream_index", 0),
            input_ordinal=item.get("input_ordinal", 0),
            output_ordinal=item.get("output_ordinal", 0),
            source_codec=str(item.get("source_codec", "")),
            start_time=item.get("start_time", 0.0),
            duration=item.get("duration"),
            expected_codec=str(item.get("expected_codec", "")),
            expected_width=item.get("expected_width"),
            expected_height=item.get("expected_height"),
            settings=StreamEncodingOverride.from_dict(item.get("settings", {})),
            use_muxer_default=item.get("use_muxer_default", False),
            fallback_reason=str(item.get("fallback_reason", "")),
            tags=normalize_tags(item.get("tags")),
            disposition=item.get("disposition", {}) if isinstance(item.get("disposition"), dict) else {},
        )


@dataclass
class ResolvedMuxStream:
    input_stream_index: int
    input_ordinal: int
    output_ordinal: int
    codec_type: str
    source_codec: str = ""
    expected_codec: str = ""
    expected_width: Optional[int] = None
    expected_height: Optional[int] = None
    action: str = "copy"
    codec: str = ""
    bitrate: str = ""
    extra_args: str = ""
    fallback_reason: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    disposition: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.input_stream_index = normalize_int(self.input_stream_index, minimum=0, default=0)
        self.input_ordinal = normalize_int(self.input_ordinal, minimum=0, default=0)
        self.output_ordinal = normalize_int(self.output_ordinal, minimum=0, default=0)
        self.codec_type = str(self.codec_type or "data").strip().lower()
        self.source_codec = str(self.source_codec or "").strip().lower()
        self.expected_codec = str(self.expected_codec or "").strip().lower()
        self.expected_width = normalize_optional_int(self.expected_width, minimum=1)
        self.expected_height = normalize_optional_int(self.expected_height, minimum=1)
        self.action = str(self.action or "copy").strip().lower()
        self.codec = str(self.codec or "").strip()
        self.bitrate = str(self.bitrate or "").strip()
        self.extra_args = str(self.extra_args or "").strip()
        self.fallback_reason = str(self.fallback_reason or "").strip()
        self.tags = normalize_tags(self.tags)
        raw_disposition = self.disposition if isinstance(self.disposition, dict) else {}
        self.disposition = {str(key): bool(value) for key, value in raw_disposition.items() if bool(value)}

    @staticmethod
    def from_dict(data: object) -> "ResolvedMuxStream":
        item = data if isinstance(data, dict) else {}
        return ResolvedMuxStream(
            input_stream_index=item.get("input_stream_index", 0),
            input_ordinal=item.get("input_ordinal", 0),
            output_ordinal=item.get("output_ordinal", 0),
            codec_type=str(item.get("codec_type", "data")),
            source_codec=str(item.get("source_codec", "")),
            expected_codec=str(item.get("expected_codec", "")),
            expected_width=item.get("expected_width"),
            expected_height=item.get("expected_height"),
            action=str(item.get("action", "copy")),
            codec=str(item.get("codec", "")),
            bitrate=str(item.get("bitrate", "")),
            extra_args=str(item.get("extra_args", "")),
            fallback_reason=str(item.get("fallback_reason", "")),
            tags=normalize_tags(item.get("tags")),
            disposition=item.get("disposition", {}) if isinstance(item.get("disposition"), dict) else {},
        )


@dataclass
class ResolvedStreamPlan:
    source_fingerprint: str
    video_tasks: List[VideoStreamTask] = field(default_factory=list)
    mux_streams: List[ResolvedMuxStream] = field(default_factory=list)
    preserve_metadata: bool = True
    preserve_chapters: bool = True
    source_tags: Dict[str, str] = field(default_factory=dict)
    chapters: List[ChapterInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.source_fingerprint = str(self.source_fingerprint or "").strip()
        self.video_tasks = [
            item if isinstance(item, VideoStreamTask) else VideoStreamTask.from_dict(item)
            for item in self.video_tasks
            if isinstance(item, (VideoStreamTask, dict))
        ]
        self.mux_streams = [
            item if isinstance(item, ResolvedMuxStream) else ResolvedMuxStream.from_dict(item)
            for item in self.mux_streams
            if isinstance(item, (ResolvedMuxStream, dict))
        ]
        self.preserve_metadata = normalize_bool(self.preserve_metadata, default=True)
        self.preserve_chapters = normalize_bool(self.preserve_chapters, default=True)
        self.source_tags = normalize_tags(self.source_tags)
        self.chapters = [
            item if isinstance(item, ChapterInfo) else ChapterInfo.from_ffprobe(item)
            for item in self.chapters
            if isinstance(item, (ChapterInfo, dict))
        ]
        self.warnings = [str(item) for item in self.warnings if str(item).strip()]

    @property
    def fallbacks(self) -> List[str]:
        result = [task.fallback_reason for task in self.video_tasks if task.fallback_reason]
        result.extend(stream.fallback_reason for stream in self.mux_streams if stream.fallback_reason)
        return result

    def fingerprint(self) -> str:
        data = asdict(self)
        data.pop("warnings", None)
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def from_dict(data: object) -> Optional["ResolvedStreamPlan"]:
        if not isinstance(data, dict):
            return None
        return ResolvedStreamPlan(
            source_fingerprint=str(data.get("source_fingerprint", "")),
            video_tasks=[
                VideoStreamTask.from_dict(item) for item in data.get("video_tasks", []) if isinstance(item, dict)
            ],
            mux_streams=[
                ResolvedMuxStream.from_dict(item) for item in data.get("mux_streams", []) if isinstance(item, dict)
            ],
            preserve_metadata=data.get("preserve_metadata", True),
            preserve_chapters=data.get("preserve_chapters", True),
            source_tags=normalize_tags(data.get("source_tags")),
            chapters=[ChapterInfo.from_ffprobe(item) for item in data.get("chapters", []) if isinstance(item, dict)],
            warnings=[str(item) for item in data.get("warnings", []) if str(item).strip()],
        )


@dataclass
class JobSpec:
    src: str
    profile_id: str
    variant_id: str
    assigned_resource_id: str = ""
    assigned_slot: int = 0
    resolved_stream_plan: Optional[ResolvedStreamPlan] = None

    def __post_init__(self) -> None:
        self.src = str(self.src or "")
        self.profile_id = str(self.profile_id or "")
        self.variant_id = str(self.variant_id or "")
        self.assigned_resource_id = str(self.assigned_resource_id or "").strip().lower()
        self.assigned_slot = normalize_int(self.assigned_slot, minimum=0, default=0)
        if self.resolved_stream_plan is not None and not isinstance(self.resolved_stream_plan, ResolvedStreamPlan):
            self.resolved_stream_plan = ResolvedStreamPlan.from_dict(self.resolved_stream_plan)


@dataclass
class FileStatus:
    path: Path
    outputs: Dict[str, bool]

    def label(self, profile: EncodeProfile) -> str:
        enabled = [variant for variant in profile.outputs if variant.enabled and variant.id in self.outputs]
        if not enabled:
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


def normalize_optional_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        return text not in {"0", "false", "no", "off", "none", "auto"}
    return bool(value)


def normalize_bool(value: object, default: bool = False) -> bool:
    normalized = normalize_optional_bool(value)
    return default if normalized is None else normalized


def normalize_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_optional_float(value: object, minimum: Optional[float] = None) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None:
        number = max(minimum, number)
    return number


def normalize_tags(value: object) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key).strip().lower(): str(item) for key, item in value.items() if str(key).strip()}


def normalize_string_list(value: object) -> List[str]:
    if isinstance(value, str):
        items = re.split(r"[,;+]", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return []
    result: List[str] = []
    for item in items:
        text = str(item or "").strip().lower()
        if text and text not in result:
            result.append(text)
    return result


def normalize_int(value: object, minimum: Optional[int] = None, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if minimum is not None:
        number = max(minimum, number)
    return number


def normalize_optional_int(value: object, minimum: Optional[int] = None) -> Optional[int]:
    if value in ("", None):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if minimum is not None:
        number = max(minimum, number)
    return number


def normalize_backend(value: object) -> str:
    backend = str(value or "").strip().lower()
    return backend if backend in BACKENDS else BACKEND_CPU


def backend_from_encoder(encoder: object) -> str:
    value = str(encoder or "").strip().lower()
    if value.endswith("_nvenc"):
        return BACKEND_NVENC
    if value.endswith("_qsv"):
        return BACKEND_QSV
    if value.endswith("_amf"):
        return BACKEND_AMF
    return BACKEND_CPU


def resource_id_for_backend(backend: str, index: int = 0) -> str:
    backend = normalize_backend(backend)
    if backend == BACKEND_CPU:
        return f"cpu:{max(0, int(index))}"
    vendor = {
        BACKEND_NVENC: "nvidia",
        BACKEND_QSV: "intel",
        BACKEND_AMF: "amd",
    }.get(backend, backend)
    return f"{vendor}:{max(0, int(index))}"


def resource_backend(resource_id: str) -> str:
    prefix = str(resource_id or "").split(":", 1)[0].lower()
    if prefix == "cpu":
        return BACKEND_CPU
    if prefix == "nvidia":
        return BACKEND_NVENC
    if prefix == "intel":
        return BACKEND_QSV
    if prefix == "amd":
        return BACKEND_AMF
    return BACKEND_CPU


def resource_index(resource_id: str) -> int:
    _prefix, sep, suffix = str(resource_id or "").partition(":")
    if not sep:
        return 0
    return normalize_int(suffix, minimum=0, default=0)


def normalize_resource_id_list(value: object) -> List[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = []
    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip().lower()
        if not text or ":" not in text:
            continue
        prefix, _, suffix = text.partition(":")
        if prefix not in {"cpu", "nvidia", "intel", "amd"}:
            continue
        resource_id = f"{prefix}:{normalize_int(suffix, minimum=0, default=0)}"
        if resource_id not in normalized:
            normalized.append(resource_id)
    return normalized


def normalize_audio_codec(value: object) -> str:
    codec = str(value or "copy").strip().lower()
    return codec if codec in AUDIO_CONTAINERS_BY_CODEC else "copy"


def audio_containers_for_codec(codec: object) -> List[str]:
    return list(AUDIO_CONTAINERS_BY_CODEC.get(normalize_audio_codec(codec), AUDIO_CONTAINERS_BY_CODEC["copy"]))


def normalize_audio_container_for_codec(codec: object, container: object) -> str:
    containers = audio_containers_for_codec(codec)
    value = normalize_container_extension(container, default="")
    return value if value in containers else containers[0]


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

    input_dir.mkdir(parents=True, exist_ok=True)
    if not has_source_template(profile.output_dir):
        profile_output_dir(profile).mkdir(parents=True, exist_ok=True)
    if not has_source_template(profile.archive_dir):
        profile_archive_dir(profile).mkdir(parents=True, exist_ok=True)
    for variant in profile.outputs:
        if variant.enabled:
            variant_input_dir(profile, variant).mkdir(parents=True, exist_ok=True)
            output_template = variant.output_dir or profile.output_dir
            if not has_source_template(output_template) and not has_source_template(variant.folder_name):
                (variant_output_dir(profile, variant) / expand_folder_template(variant.folder_name, None)).mkdir(
                    parents=True,
                    exist_ok=True,
                )


def safe_folder_name(text: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", text.strip())
    value = re.sub(r"\s+", "-", value)
    value = value.strip(".- ")
    return value or "output"


def safe_file_stem(text: str, default: str = "video") -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "-", str(text or "").strip())
    value = re.sub(r"\s+", " ", value)
    value = value.strip(".- ")
    return value or default


def normalize_container_extension(value: object, default: str = "mp4") -> str:
    candidate = str(value or "").strip().lower().lstrip(".")
    if SAFE_CONTAINER_RE.fullmatch(candidate):
        return candidate
    return default


def has_source_template(value: object) -> bool:
    text = str(value or "")
    return any(token in text for token in SOURCE_TEMPLATE_TOKENS)


def source_template_replacements(src: Path) -> Dict[str, str]:
    return {
        "${filename}": safe_folder_name(src.stem),
        "{source}": safe_folder_name(src.stem),
    }


def expand_path_template(value: object, src: Optional[Path]) -> str:
    text = str(value or "").strip()
    if src is None:
        return text
    for token, replacement in source_template_replacements(src).items():
        text = text.replace(token, replacement)
    return text


def expand_folder_template(value: object, src: Optional[Path]) -> str:
    expanded = expand_path_template(value, src)
    return safe_folder_name(expanded)


def parse_ffmpeg_args(value: object) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        lexer = shlex.shlex(text.replace("\r", " ").replace("\n", " "), posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        lexer.escape = ""
        return list(lexer)
    except ValueError as exc:
        raise ValueError(f"Invalid FFmpeg options: {exc}") from exc


def has_ffmpeg_option(args: Iterable[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def faststart_args_for_container(value: object) -> List[str]:
    container = normalize_container_extension(value)
    if container in FASTSTART_CONTAINERS:
        return ["-movflags", "+faststart"]
    return []


def faststart_args_unless_overridden(value: object, extra_args: Iterable[str]) -> List[str]:
    if has_ffmpeg_option(extra_args, "-movflags"):
        return []
    return faststart_args_for_container(value)


def default_outputs() -> List[OutputVariant]:
    return [
        OutputVariant(
            id="master_2160p",
            name="UP Convert 4K",
            folder_name="up-convert-4k",
            height=2160,
            container="mp4",
        ),
        OutputVariant(
            id="reference_original",
            name="ReEncode Original Pixel",
            folder_name="reencode-original-pixel",
            height=None,
            container="mp4",
        ),
    ]


def detect_nvenc_engine_count(_gpu_index: int = 0) -> Tuple[Optional[int], str]:
    if os.name != "nt":
        return None, "NVENC engine probing is only attempted on Windows."
    try:
        ctypes.WinDLL("nvEncodeAPI64.dll")
    except Exception as exc:
        return None, f"nvEncodeAPI64.dll unavailable: {exc}"
    # Opening an NVENC session requires a CUDA/D3D device context. The app keeps this
    # probe optional and falls back to user-tunable slots when no helper context exists.
    return None, "nvEncodeAPI64.dll loaded; encode-session caps helper unavailable, using manual slots."


def nvenc_engine_hint_from_name(name: str) -> Optional[int]:
    normalized = re.sub(r"[\s_-]+", " ", str(name or "")).strip()
    for pattern, engines in NVIDIA_NVENC_ENGINE_HINTS:
        if pattern.search(normalized):
            return engines
    return None


def fallback_cpu_resource() -> HardwareResource:
    return HardwareResource(
        id=CPU_RESOURCE_ID,
        label="CPU",
        kind="cpu",
        backend=BACKEND_CPU,
        vendor="cpu",
        index=0,
        concurrency_slots=1,
    )


def cpu_resource_from_wmi_item(item: Dict[str, object], fallback_index: int) -> Optional[HardwareResource]:
    name = str(item.get("Name") or "").strip()
    device_id = str(item.get("DeviceID") or "").strip()
    socket = str(item.get("SocketDesignation") or "").strip()
    index = fallback_index
    match = re.search(r"(\d+)", device_id)
    if match:
        index = normalize_int(match.group(1), minimum=0, default=fallback_index)
    if not name and not socket and not device_id:
        return None
    model = name or "CPU"
    socket_label = socket or device_id
    label = f"CPU {index}: {model}"
    if socket_label:
        label = f"{label} / {socket_label}"
    return HardwareResource(
        id=resource_id_for_backend(BACKEND_CPU, index),
        label=label,
        kind="cpu",
        backend=BACKEND_CPU,
        vendor="cpu",
        index=index,
        concurrency_slots=1,
    )


def cpu_resources_from_wmi_data(data: object) -> List[HardwareResource]:
    items = data if isinstance(data, list) else [data]
    resources: List[HardwareResource] = []
    seen: set[str] = set()
    for fallback_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        resource = cpu_resource_from_wmi_item(item, fallback_index)
        if resource is None or resource.id in seen:
            continue
        resources.append(resource)
        seen.add(resource.id)
    return resources or [fallback_cpu_resource()]


@lru_cache(maxsize=1)
def _detect_cpu_resources_cached(timeout: int = 5) -> Tuple[HardwareResource, ...]:
    if os.name != "nt":
        return (fallback_cpu_resource(),)
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_Processor | Select-Object DeviceID,Name,SocketDesignation | ConvertTo-Json -Compress",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            **no_window_subprocess_kwargs(),
        )
    except Exception:
        return (fallback_cpu_resource(),)
    if result.returncode != 0:
        return (fallback_cpu_resource(),)
    try:
        data = json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        return (fallback_cpu_resource(),)
    return tuple(cpu_resources_from_wmi_data(data))


def detect_cpu_resources(timeout: int = 5) -> List[HardwareResource]:
    return list(_detect_cpu_resources_cached(timeout))


def clear_cpu_resource_cache() -> None:
    _detect_cpu_resources_cached.cache_clear()


def hardware_resources_from_gpus(gpus: Optional[List[GpuInfo]]) -> List[HardwareResource]:
    resources = detect_cpu_resources()
    for gpu in gpus or []:
        backend = {
            "nvidia": BACKEND_NVENC,
            "intel": BACKEND_QSV,
            "amd": BACKEND_AMF,
        }.get(gpu.vendor, BACKEND_NVENC)
        engine_count = gpu.encoder_engines
        detection_error = gpu.detection_error
        if backend == BACKEND_NVENC and engine_count is None:
            engine_count, detection_error = detect_nvenc_engine_count(gpu.index)
            if engine_count is None:
                hinted_count = nvenc_engine_hint_from_name(gpu.name)
                if hinted_count is not None:
                    engine_count = hinted_count
                    detection_error = (
                        f"{detection_error} Model hint selected {hinted_count} NVENC engine(s); "
                        "encoder availability is still checked before encoding."
                    ).strip()
        slots = engine_count or 1
        resources.append(
            HardwareResource(
                id=resource_id_for_backend(backend, gpu.index),
                label=f"GPU {gpu.index}: {gpu.name}",
                kind="gpu",
                backend=backend,
                vendor=gpu.vendor,
                index=gpu.index,
                concurrency_slots=slots,
                detected_encoder_engines=engine_count,
                detection_error=detection_error,
            )
        )
    return resources


def normalize_hardware_resources(
    resources: Iterable[HardwareResource],
    gpus: Optional[List[GpuInfo]],
) -> List[HardwareResource]:
    by_id = {resource.id: resource for resource in hardware_resources_from_gpus(gpus)}
    for resource in resources:
        if not resource.id:
            continue
        if resource.id in by_id:
            detected = by_id[resource.id]
            detected.concurrency_slots = resource.concurrency_slots
            by_id[resource.id] = detected
            continue
        if resource.id not in by_id and resource.detection_error in STALE_AUTO_RESOURCE_ERRORS:
            continue
        if (
            resource.id not in by_id
            and resource_backend(resource.id) == BACKEND_NVENC
            and not resource.detection_error.startswith(MANUAL_RESOURCE_ERROR_PREFIX)
        ):
            continue
        by_id[resource.id] = resource
    if CPU_RESOURCE_ID not in by_id:
        by_id[CPU_RESOURCE_ID] = fallback_cpu_resource()
    return sorted(by_id.values(), key=lambda item: (item.kind != "cpu", item.vendor, item.index, item.id))


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
            **no_window_subprocess_kwargs(),
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


def gpu_vendor_from_wmi_item(item: Dict[str, object]) -> str:
    haystack = " ".join(str(item.get(key) or "") for key in ("Name", "PNPDeviceID", "AdapterCompatibility")).lower()
    if "10de" in haystack or "nvidia" in haystack:
        return "nvidia"
    if "8086" in haystack or "intel" in haystack:
        return "intel"
    if "1002" in haystack or "amd" in haystack or "advanced micro" in haystack or "radeon" in haystack:
        return "amd"
    return ""


def gpus_from_wmi_data(data: object, existing: Optional[Iterable[GpuInfo]] = None) -> List[GpuInfo]:
    items = data if isinstance(data, list) else [data]
    counts: Dict[str, int] = {"nvidia": 0, "intel": 0, "amd": 0}
    existing_keys = {
        (gpu.vendor, re.sub(r"\s+", " ", gpu.name).strip().lower()) for gpu in (existing or []) if gpu.name
    }
    gpus: List[GpuInfo] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        vendor = gpu_vendor_from_wmi_item(item)
        if not name or vendor not in counts:
            continue
        key = (vendor, re.sub(r"\s+", " ", name).strip().lower())
        if key in existing_keys:
            counts[vendor] += 1
            continue
        index = counts[vendor]
        counts[vendor] += 1
        gpus.append(GpuInfo(index=index, name=name, vendor=vendor))
    return gpus


def detect_wmi_gpus(timeout: int = 5, existing: Optional[Iterable[GpuInfo]] = None) -> List[GpuInfo]:
    if os.name != "nt":
        return []
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,PNPDeviceID,AdapterCompatibility | ConvertTo-Json -Compress"
        ),
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            **no_window_subprocess_kwargs(),
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    try:
        data = json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        return []
    return gpus_from_wmi_data(data, existing)


def detect_gpus(timeout: int = 5) -> List[GpuInfo]:
    nvidia = detect_nvidia_gpus(timeout)
    return nvidia + detect_wmi_gpus(timeout, existing=nvidia)


def normalize_profile_gpu(profile: EncodeProfile, gpus: Optional[List[GpuInfo]]) -> EncodeProfile:
    profile.hardware_resources = normalize_hardware_resources(profile.hardware_resources, gpus)
    resource_by_id = {resource.id: resource for resource in profile.hardware_resources}
    if not profile.resource_ids:
        if profile.use_gpu:
            profile.resource_ids = [resource_id_for_backend(BACKEND_NVENC, profile.gpu_index)]
        else:
            profile.resource_ids = [CPU_RESOURCE_ID]
    profile.resource_ids = [resource_id for resource_id in profile.resource_ids if resource_id in resource_by_id]
    if not profile.resource_ids:
        profile.resource_ids = [CPU_RESOURCE_ID]

    gpu_by_index = {gpu.index: gpu for gpu in gpus or []}
    if profile.use_gpu:
        gpu = gpu_by_index.get(profile.gpu_index)
        if gpu is not None:
            profile.gpu_name = gpu.name
        else:
            profile.use_gpu = False
            profile.gpu_index = 0
            profile.gpu_name = ""

    for variant in profile.outputs:
        if not variant.backend:
            if profile.use_gpu:
                variant.backend = BACKEND_NVENC
                variant.ffmpeg_encoder = variant.codec or profile.codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_NVENC]
                variant.resource_ids = [resource_id_for_backend(BACKEND_NVENC, profile.gpu_index)]
            else:
                variant.backend = BACKEND_CPU
                variant.ffmpeg_encoder = (
                    variant.cpu_codec or profile.cpu_codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU]
                )
                variant.resource_ids = [CPU_RESOURCE_ID]
        variant.backend = normalize_backend(variant.backend)
        if not variant.ffmpeg_encoder:
            variant.ffmpeg_encoder = DEFAULT_ENCODER_BY_BACKEND[variant.backend]
        valid_resources = [
            resource_id
            for resource_id in variant.resource_ids
            if resource_id in resource_by_id and resource_backend(resource_id) == variant.backend
        ]
        if not valid_resources:
            if variant.use_gpu is True:
                candidate = resource_id_for_backend(
                    BACKEND_NVENC, variant.gpu_index if variant.gpu_index is not None else profile.gpu_index
                )
                if candidate in resource_by_id:
                    valid_resources = [candidate]
            elif variant.backend == BACKEND_CPU and CPU_RESOURCE_ID in resource_by_id:
                valid_resources = [CPU_RESOURCE_ID]
        if not valid_resources:
            variant.backend = BACKEND_CPU
            variant.ffmpeg_encoder = variant.cpu_codec or DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU]
            variant.use_gpu = False
            variant.gpu_index = 0
            variant.gpu_name = ""
            variant.resource_ids = [CPU_RESOURCE_ID]
        else:
            variant.resource_ids = valid_resources
            variant.use_gpu = variant.backend != BACKEND_CPU
            if variant.backend == BACKEND_NVENC:
                gpu_index = resource_index(valid_resources[0])
                gpu = gpu_by_index.get(gpu_index)
                variant.gpu_index = gpu_index
                variant.gpu_name = gpu.name if gpu else ""
                if not variant.codec:
                    variant.codec = variant.ffmpeg_encoder or DEFAULT_ENCODER_BY_BACKEND[BACKEND_NVENC]
            elif variant.backend == BACKEND_CPU:
                variant.gpu_index = 0
                variant.gpu_name = ""
                if not variant.cpu_codec:
                    variant.cpu_codec = variant.ffmpeg_encoder or DEFAULT_ENCODER_BY_BACKEND[BACKEND_CPU]
    return profile


def default_profile(paths: AppPaths, gpus: Optional[List[GpuInfo]] = None) -> EncodeProfile:
    detected = detect_gpus() if gpus is None else gpus
    gpu = next((item for item in detected if item.vendor == "nvidia"), None)
    hardware_resources = hardware_resources_from_gpus(detected)
    resource_ids = [
        resource_id_for_backend(
            {
                "nvidia": BACKEND_NVENC,
                "intel": BACKEND_QSV,
                "amd": BACKEND_AMF,
            }.get(gpu_info.vendor, BACKEND_NVENC),
            gpu_info.index,
        )
        for gpu_info in detected
    ] or [CPU_RESOURCE_ID]
    output_backend = resource_backend(resource_ids[0])
    output_resource_ids = [
        resource_id for resource_id in resource_ids if resource_backend(resource_id) == output_backend
    ]
    outputs = default_outputs()
    for output in outputs:
        output.backend = output_backend
        output.ffmpeg_encoder = DEFAULT_ENCODER_BY_BACKEND[output_backend]
        output.resource_ids = list(output_resource_ids)
        output.use_gpu = output_backend != BACKEND_CPU
        if output_backend == BACKEND_NVENC:
            output.codec = output.ffmpeg_encoder
            output.gpu_index = resource_index(output_resource_ids[0]) if output_resource_ids else 0
        elif output_backend == BACKEND_CPU:
            output.cpu_codec = output.ffmpeg_encoder
    return EncodeProfile(
        id="default",
        input_dir=str(paths.base_dir / "Incoming"),
        output_dir=str(paths.base_dir / "output" / "{source}"),
        archive_dir=str(paths.base_dir / "output" / "{source}"),
        max_parallel_jobs=2 if gpu else 1,
        segment_minutes=10,
        use_gpu=gpu is not None,
        gpu_index=gpu.index if gpu else 0,
        gpu_name=gpu.name if gpu else "",
        codec="hevc_nvenc",
        cpu_codec="libx264",
        cpu_preset="medium",
        cpu_tune="none",
        cq_value=15 if gpu else 23,
        bitrate="25000k" if gpu else "8000k",
        maxrate="40000k" if gpu else "12000k",
        bufsize="80000k" if gpu else "24000k",
        resource_ids=resource_ids,
        hardware_resources=hardware_resources,
        outputs=outputs,
    )


def read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def stream_override_store_path(paths: AppPaths) -> Path:
    return paths.tmp_dir / "stream_overrides.json"


def stream_override_key(src: Path, profile_id: str, variant_id: str) -> str:
    value = f"{source_fingerprint(src)}\0{profile_id}\0{variant_id}"
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def load_stream_override_store(paths: AppPaths) -> Dict[str, Any]:
    data = read_json_object(stream_override_store_path(paths))
    if not data or data.get("version") != 1 or not isinstance(data.get("entries"), dict):
        return {"version": 1, "entries": {}}
    return data


def load_stream_overrides(
    paths: AppPaths,
    src: Path,
    profile_id: str,
    variant_id: str,
) -> Dict[int, StreamEncodingOverride]:
    store = load_stream_override_store(paths)
    entry = store["entries"].get(stream_override_key(src, profile_id, variant_id), {})
    if not isinstance(entry, dict) or entry.get("source_fingerprint") != source_fingerprint(src):
        return {}
    raw = entry.get("overrides", {})
    if not isinstance(raw, dict):
        return {}
    result: Dict[int, StreamEncodingOverride] = {}
    for key, value in raw.items():
        try:
            stream_index = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            result[stream_index] = StreamEncodingOverride.from_dict(value)
    return result


def save_stream_overrides(
    paths: AppPaths,
    src: Path,
    profile_id: str,
    variant_id: str,
    overrides: Dict[int, StreamEncodingOverride],
) -> None:
    ensure_dirs(paths)
    store = load_stream_override_store(paths)
    key = stream_override_key(src, profile_id, variant_id)
    if overrides:
        store["entries"][key] = {
            "source_path": str(src.resolve()),
            "source_fingerprint": source_fingerprint(src),
            "profile_id": profile_id,
            "variant_id": variant_id,
            "overrides": {str(index): asdict(value) for index, value in overrides.items()},
        }
    else:
        store["entries"].pop(key, None)
    path = stream_override_store_path(paths)
    tmp_file = path.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(path)


def remove_stream_overrides_for_source(paths: AppPaths, src: Path) -> None:
    store = load_stream_override_store(paths)
    resolved = str(src.resolve())
    entries = store["entries"]
    keys = [key for key, entry in entries.items() if isinstance(entry, dict) and entry.get("source_path") == resolved]
    if not keys:
        return
    for key in keys:
        entries.pop(key, None)
    path = stream_override_store_path(paths)
    tmp_file = path.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(path)


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
    existing = read_json_object(paths.config_file) if paths.config_file.exists() else None
    if existing and normalize_int(existing.get("version", 0), minimum=0, default=0) < 3:
        backup = paths.config_file.with_name(f"{paths.config_file.stem}.v2.json.bak")
        if not backup.exists():
            shutil.copy2(paths.config_file, backup)
    data = {
        "version": 3,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profiles": [profile_to_dict(profile) for profile in profiles],
    }
    tmp_file = paths.config_file.with_suffix(".json.tmp")
    tmp_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(paths.config_file)


def output_variant_to_dict(variant: OutputVariant) -> Dict[str, Any]:
    data = asdict(variant)
    if variant.backend and variant.ffmpeg_encoder:
        data.pop("use_gpu", None)
        data.pop("gpu_index", None)
        data.pop("gpu_name", None)
    return data


def profile_to_dict(profile: EncodeProfile) -> Dict[str, Any]:
    data = asdict(profile)
    for key in DEPRECATED_PROFILE_FIELDS:
        data.pop(key, None)
    data["outputs"] = [output_variant_to_dict(variant) for variant in profile.outputs]
    data["hardware_resources"] = [asdict(resource) for resource in profile.hardware_resources]
    return data


def variant_by_id(profile: EncodeProfile, variant_id: str) -> OutputVariant:
    for variant in profile.outputs:
        if variant.id == variant_id:
            return variant
    raise ValueError(f"Unknown output variant: {variant_id}")


def profile_input_dir(profile: EncodeProfile) -> Path:
    return Path(profile.input_dir).expanduser().resolve()


def profile_output_dir(profile: EncodeProfile, src: Optional[Path] = None) -> Path:
    return Path(expand_path_template(profile.output_dir, src)).expanduser().resolve()


def profile_archive_dir(profile: EncodeProfile, src: Optional[Path] = None) -> Path:
    return Path(expand_path_template(profile.archive_dir, src)).expanduser().resolve()


def variant_input_dir(profile: EncodeProfile, variant: OutputVariant) -> Path:
    value = str(getattr(variant, "input_dir", "") or "").strip()
    return Path(value).expanduser().resolve() if value else profile_input_dir(profile)


def variant_output_dir(profile: EncodeProfile, variant: OutputVariant, src: Optional[Path] = None) -> Path:
    value = str(getattr(variant, "output_dir", "") or "").strip()
    return Path(expand_path_template(value, src)).expanduser().resolve() if value else profile_output_dir(profile, src)


def variant_segment_minutes(profile: EncodeProfile, variant: OutputVariant) -> int:
    value = getattr(variant, "segment_minutes", None)
    if value not in (None, ""):
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            pass
    return max(1, int(getattr(profile, "segment_minutes", 10) or 10))


def variant_accepts_source(profile: EncodeProfile, variant: OutputVariant, src: Path) -> bool:
    try:
        return src.parent.resolve() == variant_input_dir(profile, variant)
    except OSError:
        return False


def output_file_stem_for(profile: EncodeProfile, src: Path, variant: OutputVariant) -> str:
    height = "" if variant.height is None else str(variant.height)
    replacements = {
        "${filename}": src.stem,
        "{source}": src.stem,
        "{profile}": variant.name,
        "{profile_id}": variant.id,
        "{folder}": variant.folder_name,
        "{height}": height,
    }
    template = variant.filename_template.strip() or "{source}"
    value = template
    for token, replacement in replacements.items():
        value = value.replace(token, replacement)
    return safe_file_stem(value, default=safe_file_stem(src.stem))


def output_path_for(profile: EncodeProfile, src: Path, variant: OutputVariant) -> Path:
    container = normalize_container_extension(variant.container)
    folder_name = expand_folder_template(variant.folder_name, src)
    return (
        variant_output_dir(profile, variant, src)
        / folder_name
        / f"{output_file_stem_for(profile, src, variant)}.{container}"
    )


def scan_profile_files(profile: EncodeProfile) -> List[FileStatus]:
    if missing_profile_dirs(profile):
        return []

    outputs_by_path: Dict[Path, Dict[str, bool]] = {}
    for variant in profile.outputs:
        if not variant.enabled:
            continue
        input_dir = variant_input_dir(profile, variant)
        if not input_dir.exists():
            continue
        for path in sorted(input_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
                continue
            outputs = outputs_by_path.setdefault(path.resolve(), {})
            outputs[variant.id] = output_path_for(profile, path, variant).exists()
    return [
        FileStatus(path=path, outputs=outputs)
        for path, outputs in sorted(outputs_by_path.items(), key=lambda item: str(item[0]).lower())
    ]


def variant_resource_ids(profile: EncodeProfile, variant: OutputVariant) -> List[str]:
    backend = effective_backend(profile, variant)
    resource_by_id = {resource.id: resource for resource in profile.hardware_resources}
    selected = [
        resource_id
        for resource_id in variant.resource_ids
        if resource_id in resource_by_id and resource_backend(resource_id) == backend
    ]
    if selected:
        return selected
    profile_selected = [
        resource_id
        for resource_id in profile.resource_ids
        if resource_id in resource_by_id and resource_backend(resource_id) == backend
    ]
    if profile_selected:
        return profile_selected
    if profile.hardware_resources:
        return []
    fallback = CPU_RESOURCE_ID if backend == BACKEND_CPU else resource_id_for_backend(backend, 0)
    return [fallback]


def build_job_specs(profile: EncodeProfile, files: Iterable[Path]) -> List[JobSpec]:
    specs: List[JobSpec] = []
    next_resource_index: Dict[str, int] = {}
    for src in files:
        src = Path(src)
        for variant in profile.outputs:
            if not variant.enabled:
                continue
            if not variant_accepts_source(profile, variant, src):
                continue
            if not output_path_for(profile, src, variant).exists():
                resources = variant_resource_ids(profile, variant)
                next_index = next_resource_index.get(variant.id, 0)
                assigned = resources[next_index % len(resources)] if resources else ""
                next_resource_index[variant.id] = next_index + 1
                specs.append(
                    JobSpec(
                        src=str(src),
                        profile_id=profile.id,
                        variant_id=variant.id,
                        assigned_resource_id=assigned,
                    )
                )
    return specs


def duplicate_output_targets_for_outputs(outputs: Iterable[OutputVariant]) -> List[str]:
    seen: set[Tuple[str, str, str]] = set()
    duplicates: List[str] = []
    for variant in outputs:
        if not variant.enabled:
            continue
        key = (
            variant.folder_name.lower(),
            normalize_container_extension(variant.container),
            (variant.filename_template or "{source}").strip().lower(),
        )
        if key in seen:
            duplicates.append(f"{variant.folder_name}/{key[2]}.{key[1]}")
        else:
            seen.add(key)
    return duplicates


def duplicate_output_targets(profile: EncodeProfile) -> List[str]:
    return duplicate_output_targets_for_outputs(profile.outputs)


def variant_setting(profile: EncodeProfile, variant: OutputVariant, name: str, default: object = "") -> object:
    value = getattr(variant, name, None)
    if value not in (None, ""):
        return value
    return getattr(profile, name, default)


def variant_use_gpu(profile: EncodeProfile, variant: OutputVariant) -> bool:
    if variant.backend:
        return variant.backend != BACKEND_CPU
    if variant.use_gpu is not None:
        return variant.use_gpu
    return profile.use_gpu


def effective_backend(profile: EncodeProfile, variant: OutputVariant) -> str:
    if variant.backend:
        return normalize_backend(variant.backend)
    return BACKEND_NVENC if variant_use_gpu(profile, variant) else BACKEND_CPU


def variant_gpu_index(profile: EncodeProfile, variant: OutputVariant, resource_id: str = "") -> int:
    if resource_id:
        return resource_index(resource_id)
    resources = variant_resource_ids(profile, variant)
    for item in resources:
        if resource_backend(item) == BACKEND_NVENC:
            return resource_index(item)
    if variant.gpu_index is not None:
        return max(0, variant.gpu_index)
    return max(0, profile.gpu_index)


def variant_audio_codec(profile: EncodeProfile, variant: OutputVariant) -> str:
    value = str(variant_setting(profile, variant, "audio_codec", "copy") or "").strip()
    return normalize_audio_codec(value)


def variant_audio_container(profile: EncodeProfile, variant: OutputVariant) -> str:
    configured = str(variant.audio_container or "").strip()
    return normalize_audio_container_for_codec(variant_audio_codec(profile, variant), configured)


def validate_stream_extra_args(value: object) -> List[str]:
    errors: List[str] = []
    try:
        tokens = parse_ffmpeg_args(value)
    except ValueError as exc:
        return [str(exc)]
    for token in tokens:
        if not token.startswith("-") or token == "-":
            continue
        base = token.split(":", 1)[0]
        if base not in STREAM_SCOPED_EXTRA_OPTIONS:
            errors.append(f"stream別追加引数では使用できません: {token}")
    return errors


def scope_stream_extra_args(value: object, codec_type: str, output_ordinal: int) -> List[str]:
    errors = validate_stream_extra_args(value)
    if errors:
        raise ValueError("; ".join(errors))
    stream_letter = {"video": "v", "audio": "a", "subtitle": "s"}.get(codec_type, codec_type[:1])
    scoped: List[str] = []
    for token in parse_ffmpeg_args(value):
        if token.startswith("-") and token != "-":
            base = token.split(":", 1)[0]
            token = (
                f"-metadata:s:{stream_letter}:{output_ordinal}"
                if base == "-metadata"
                else f"{base}:{stream_letter}:{output_ordinal}"
            )
        scoped.append(token)
    return scoped


def validate_stream_rules(variant: OutputVariant) -> List[str]:
    errors: List[str] = []
    ids: set[str] = set()
    fallback_kinds: set[tuple[str, Optional[bool]]] = set()
    for rule in variant.stream_rules:
        if rule.id in ids:
            errors.append(f"stream rule IDが重複しています: {rule.id}")
        ids.add(rule.id)
        selector = rule.selector
        if (
            selector.ordinal is None
            and not selector.codec_names
            and not selector.languages
            and not selector.title_contains
            and not selector.dispositions
        ):
            if selector.kind == "video" and selector.attached_pic is None:
                fallback_kinds.update({("video", False), ("video", True)})
            else:
                fallback_kinds.add((selector.kind, selector.attached_pic if selector.kind == "video" else None))
        errors.extend(f"{rule.name}: {message}" for message in validate_stream_extra_args(rule.encoding.extra_args))
        if rule.encoding.resource_ids and rule.encoding.backend:
            if any(
                resource_backend(resource_id) != rule.encoding.backend for resource_id in rule.encoding.resource_ids
            ):
                errors.append(f"{rule.name}: backendとresource_idsが一致しません")
    required = {
        ("video", False),
        ("video", True),
        ("audio", None),
        ("subtitle", None),
        ("attachment", None),
        ("data", None),
    }
    for kind, attached in sorted(required - fallback_kinds, key=lambda item: (item[0], str(item[1]))):
        label = "cover" if kind == "video" and attached else kind
        errors.append(f"{label}のfallback ruleがありません")
    return errors


def matching_stream_rule(variant: OutputVariant, stream: MediaStream) -> Optional[StreamRule]:
    return next((rule for rule in variant.stream_rules if rule.selector.matches(stream)), None)


def resolved_video_settings(
    profile: EncodeProfile,
    variant: OutputVariant,
    configured: StreamEncodingOverride,
) -> StreamEncodingOverride:
    backend = configured.backend or variant.backend or resource_backend(variant_resource_ids(profile, variant)[0])
    encoder = configured.ffmpeg_encoder or encoder_codec(profile, variant)
    resources = configured.resource_ids or [
        resource_id
        for resource_id in variant_resource_ids(profile, variant)
        if resource_backend(resource_id) == backend
    ]
    if not resources:
        resources = [CPU_RESOURCE_ID] if backend == BACKEND_CPU else [resource_id_for_backend(backend, 0)]
    if configured.action in {"copy", "auto"}:
        backend = BACKEND_CPU
        encoder = ""
        resources = [CPU_RESOURCE_ID]
    preset_default = (
        variant_setting(profile, variant, "cpu_preset", profile.cpu_preset)
        if backend == BACKEND_CPU
        else variant_setting(profile, variant, "preset", profile.preset)
    )
    tune_default = (
        variant_setting(profile, variant, "cpu_tune", profile.cpu_tune)
        if backend == BACKEND_CPU
        else variant_setting(profile, variant, "tune", profile.tune)
    )
    return StreamEncodingOverride(
        action=configured.action if configured.action in {"copy", "auto"} else "transcode",
        backend=backend,
        ffmpeg_encoder=encoder,
        resource_ids=resources,
        height=configured.height if configured.height is not None else variant.height,
        rate_mode=configured.rate_mode or str(variant_setting(profile, variant, "rate_mode", profile.rate_mode)),
        cq_value=(
            configured.cq_value
            if configured.cq_value is not None
            else normalize_optional_int(variant_setting(profile, variant, "cq_value", profile.cq_value))
        ),
        bitrate=configured.bitrate or str(variant_setting(profile, variant, "bitrate", profile.bitrate)),
        maxrate=configured.maxrate or str(variant_setting(profile, variant, "maxrate", profile.maxrate)),
        bufsize=configured.bufsize or str(variant_setting(profile, variant, "bufsize", profile.bufsize)),
        preset=configured.preset or str(preset_default),
        tune=configured.tune or str(tune_default),
        pix_fmt=configured.pix_fmt or output_pix_fmt(profile, variant, resources[0] if resources else ""),
        scale_flags=configured.scale_flags
        or str(variant_setting(profile, variant, "scale_flags", profile.scale_flags)),
        codec=configured.codec,
        extra_args=configured.extra_args,
    )


def resolve_stream_plan(
    profile: EncodeProfile,
    variant: OutputVariant,
    probe: MediaProbe,
    file_overrides: Optional[Dict[int, StreamEncodingOverride]] = None,
) -> ResolvedStreamPlan:
    overrides = file_overrides or {}
    video_tasks: List[VideoStreamTask] = []
    mux_streams: List[ResolvedMuxStream] = []
    output_ordinals: Dict[str, int] = {kind: 0 for kind in STREAM_KINDS}
    warnings: List[str] = []

    for stream in probe.streams:
        rule = matching_stream_rule(variant, stream)
        if rule is None:
            warnings.append(f"stream {stream.index} ({stream.codec_type}) に一致するruleがないため除外しました")
            continue
        configured = overrides.get(stream.index, rule.encoding)
        action = configured.action
        if action == "exclude":
            continue

        if stream.codec_type == "video" and not stream.attached_pic:
            settings = resolved_video_settings(profile, variant, configured)
            video_tasks.append(
                VideoStreamTask(
                    input_stream_index=stream.index,
                    input_ordinal=stream.ordinal,
                    output_ordinal=len(video_tasks),
                    source_codec=stream.codec_name,
                    start_time=stream.start_time,
                    duration=stream.duration,
                    settings=settings,
                    tags=stream.tags,
                    disposition=stream.disposition,
                )
            )
            output_ordinals["video"] += 1
            continue

        codec = configured.codec
        if stream.codec_type == "audio" and not codec:
            codec = variant_audio_codec(profile, variant)
        elif action == "auto" and not codec:
            codec = "copy"
        mux_action = action
        if mux_action == "transcode" and not codec:
            mux_action = "auto"
        mux_streams.append(
            ResolvedMuxStream(
                input_stream_index=stream.index,
                input_ordinal=stream.ordinal,
                output_ordinal=output_ordinals.get(stream.codec_type, 0),
                codec_type=stream.codec_type,
                source_codec=stream.codec_name,
                expected_width=stream.width if stream.codec_type == "video" else None,
                expected_height=stream.height if stream.codec_type == "video" else None,
                action=mux_action,
                codec=codec,
                bitrate=(
                    configured.bitrate
                    or (
                        str(variant_setting(profile, variant, "audio_bitrate", "") or "").strip()
                        if stream.codec_type == "audio"
                        else ""
                    )
                ),
                extra_args=configured.extra_args,
                tags=stream.tags,
                disposition=stream.disposition,
            )
        )
        output_ordinals[stream.codec_type] = output_ordinals.get(stream.codec_type, 0) + 1

    if not video_tasks:
        raise ValueError("通常Video streamが1つも選択されていません")
    cover_ordinal = len(video_tasks)
    for stream in mux_streams:
        if stream.codec_type == "video":
            stream.output_ordinal = cover_ordinal
            cover_ordinal += 1
    return ResolvedStreamPlan(
        source_fingerprint=source_fingerprint(Path(probe.path)),
        video_tasks=video_tasks,
        mux_streams=mux_streams,
        preserve_metadata=variant.preserve_metadata,
        preserve_chapters=variant.preserve_chapters,
        source_tags=probe.tags,
        chapters=probe.chapters,
        warnings=warnings,
    )


def is_nvenc_codec(codec: str) -> bool:
    return codec.lower().endswith("_nvenc")


def is_qsv_codec(codec: str) -> bool:
    return codec.lower().endswith("_qsv")


def is_amf_codec(codec: str) -> bool:
    return codec.lower().endswith("_amf")


def is_hardware_codec(codec: str) -> bool:
    return is_nvenc_codec(codec) or is_qsv_codec(codec) or is_amf_codec(codec)


def encoder_codec(profile: EncodeProfile, variant: Optional[OutputVariant] = None) -> str:
    if variant is not None:
        if variant.ffmpeg_encoder:
            return variant.ffmpeg_encoder
        backend = effective_backend(profile, variant)
        if backend == BACKEND_NVENC:
            return str(variant_setting(profile, variant, "codec", profile.codec) or "hevc_nvenc")
        if backend == BACKEND_CPU:
            return str(variant_setting(profile, variant, "cpu_codec", profile.cpu_codec) or "libx264")
        return DEFAULT_ENCODER_BY_BACKEND.get(backend, "libx264")
    if profile.use_gpu:
        return profile.codec
    return profile.cpu_codec


def output_pix_fmt(profile: EncodeProfile, variant: Optional[OutputVariant] = None, resource_id: str = "") -> str:
    if is_nvenc_codec(encoder_codec(profile, variant)):
        if variant is not None:
            return str(variant_setting(profile, variant, "pix_fmt", profile.pix_fmt) or "nv12")
        return profile.pix_fmt
    return "yuv420p"


def missing_rate_fields(profile: EncodeProfile, variant: Optional[OutputVariant] = None) -> List[str]:
    if variant is None:
        mode_name = profile.rate_mode.upper()
        bitrate = profile.bitrate
        maxrate = profile.maxrate
        bufsize = profile.bufsize
    else:
        mode_name = str(variant_setting(profile, variant, "rate_mode", profile.rate_mode) or "CQ").upper()
        bitrate = str(variant_setting(profile, variant, "bitrate", profile.bitrate) or "")
        maxrate = str(variant_setting(profile, variant, "maxrate", profile.maxrate) or "")
        bufsize = str(variant_setting(profile, variant, "bufsize", profile.bufsize) or "")
    required: List[Tuple[str, str]] = []
    if mode_name == "VBR":
        required = [("bitrate", bitrate), ("maxrate", maxrate), ("bufsize", bufsize)]
    elif mode_name == "ABR":
        required = [("bitrate", bitrate)]
    elif mode_name == "CBR":
        required = [("bitrate", bitrate), ("bufsize", bufsize)]
    return [name for name, value in required if not str(value or "").strip()]


def validate_rate_settings(profile: EncodeProfile, variant: Optional[OutputVariant] = None) -> None:
    missing = missing_rate_fields(profile, variant)
    if missing:
        mode = (
            profile.rate_mode
            if variant is None
            else str(variant_setting(profile, variant, "rate_mode", profile.rate_mode))
        )
        raise ValueError(f"{mode} requires: {', '.join(missing)}")


def build_video_encoder_args(
    profile: EncodeProfile,
    variant: Optional[OutputVariant] = None,
    resource_id: str = "",
) -> List[str]:
    validate_rate_settings(profile, variant)
    codec = encoder_codec(profile, variant)
    cmd: List[str] = ["-c:v", codec]

    if is_nvenc_codec(codec):
        gpu_index = (
            variant_gpu_index(profile, variant, resource_id) if variant is not None else max(profile.gpu_index, 0)
        )
        preset = str(
            variant_setting(profile, variant, "preset", profile.preset) if variant is not None else profile.preset
        )
        tune = str(variant_setting(profile, variant, "tune", profile.tune) if variant is not None else profile.tune)
        cmd += ["-gpu", str(gpu_index)]
        cmd += ["-preset:v", preset]
        if tune != "none":
            cmd += ["-tune:v", tune]
        if variant is not None:
            split_mode = str(getattr(variant, "split_encode_mode", "") or "").strip()
            if split_mode and split_mode not in {"auto", "default"} and codec.lower() in {"hevc_nvenc", "av1_nvenc"}:
                cmd += ["-split_encode_mode", split_mode]
    elif not is_hardware_codec(codec):
        cpu_preset = str(
            variant_setting(profile, variant, "cpu_preset", profile.cpu_preset)
            if variant is not None
            else profile.cpu_preset
        )
        cpu_tune = str(
            variant_setting(profile, variant, "cpu_tune", profile.cpu_tune) if variant is not None else profile.cpu_tune
        )
        cmd += ["-preset:v", cpu_preset]
        if cpu_tune != "none":
            cmd += ["-tune:v", cpu_tune]

    mode_name = str(
        variant_setting(profile, variant, "rate_mode", profile.rate_mode) if variant is not None else profile.rate_mode
    ).upper()
    cq_value = int(
        variant_setting(profile, variant, "cq_value", profile.cq_value) if variant is not None else profile.cq_value
    )
    bitrate = str(
        variant_setting(profile, variant, "bitrate", profile.bitrate) if variant is not None else profile.bitrate
    )
    maxrate = str(
        variant_setting(profile, variant, "maxrate", profile.maxrate) if variant is not None else profile.maxrate
    )
    bufsize = str(
        variant_setting(profile, variant, "bufsize", profile.bufsize) if variant is not None else profile.bufsize
    )
    if is_nvenc_codec(codec):
        if mode_name == "CQ":
            cmd += ["-b:v", "0", "-cq:v", str(cq_value)]
        elif mode_name == "VBR":
            cmd += [
                "-rc:v",
                "vbr",
                "-b:v",
                bitrate,
                "-maxrate:v",
                maxrate,
                "-bufsize:v",
                bufsize,
                "-cq:v",
                str(cq_value),
            ]
        elif mode_name == "ABR":
            cmd += ["-b:v", bitrate]
        elif mode_name == "CBR":
            cmd += [
                "-rc:v",
                "cbr",
                "-b:v",
                bitrate,
                "-maxrate:v",
                bitrate,
                "-bufsize:v",
                bufsize,
            ]
        else:
            raise ValueError(f"Unknown rate mode: {mode_name}")
    elif is_hardware_codec(codec):
        if mode_name == "CQ":
            raise ValueError(
                f"{codec} does not use the shared CQ/CRF control. Use VBR, ABR, CBR, or backend-specific extra args."
            )
        if mode_name in {"VBR", "ABR"}:
            cmd += ["-b:v", bitrate]
            if mode_name == "VBR":
                cmd += ["-maxrate:v", maxrate, "-bufsize:v", bufsize]
        elif mode_name == "CBR":
            cmd += ["-b:v", bitrate, "-maxrate:v", bitrate, "-bufsize:v", bufsize]
        else:
            raise ValueError(f"Unknown rate mode: {mode_name}")
    else:
        if mode_name == "CQ":
            cmd += ["-crf", str(cq_value)]
        elif mode_name in {"VBR", "ABR"}:
            cmd += ["-b:v", bitrate]
            if mode_name == "VBR":
                cmd += ["-maxrate:v", maxrate, "-bufsize:v", bufsize]
        elif mode_name == "CBR":
            cmd += [
                "-b:v",
                bitrate,
                "-maxrate:v",
                bitrate,
                "-bufsize:v",
                bufsize,
            ]
        else:
            raise ValueError(f"Unknown rate mode: {mode_name}")

    return cmd


def build_ffmpeg_command(
    ffmpeg_path: Path,
    src: Path,
    tmp_out: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
    start_seconds: Optional[float] = None,
    duration_seconds: Optional[float] = None,
    resource_id: str = "",
    input_stream_index: Optional[int] = None,
    omit_video_encoder: bool = False,
    copy_video_stream: bool = False,
    pix_fmt_override: str = "",
) -> List[str]:
    cmd: List[str] = [
        str(ffmpeg_path),
        "-hide_banner",
        "-stats_period",
        "1",
    ]
    extra_input_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_input_args", ""))
    extra_video_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_video_args", ""))
    extra_output_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_output_args", ""))

    # Segment resume prioritizes fast input seeking; exact boundaries are handled at segment granularity.
    if start_seconds is not None and start_seconds > 0:
        cmd += ["-ss", format_seconds(start_seconds)]
    if duration_seconds is not None and duration_seconds > 0:
        cmd += ["-t", format_seconds(duration_seconds)]
    cmd += extra_input_args
    cmd += [
        "-i",
        str(src),
        "-y",
        "-map",
        f"0:{input_stream_index}" if input_stream_index is not None else "0:v:0",
        "-an",
    ]
    if copy_video_stream:
        cmd += ["-c:v", "copy"]
    elif not omit_video_encoder:
        cmd += ["-pix_fmt", pix_fmt_override or output_pix_fmt(profile, variant, resource_id)]
        cmd += build_video_encoder_args(profile, variant, resource_id)

    if not copy_video_stream and variant.height is not None and variant.height > 0:
        scale_flags = str(
            variant_setting(profile, variant, "scale_flags", profile.scale_flags) or "lanczos+accurate_rnd"
        )
        scale = f"scale=-2:{variant.height}:flags={scale_flags}"
        cmd += ["-vf", scale]

    if not omit_video_encoder and not copy_video_stream:
        cmd += extra_video_args
    cmd += faststart_args_unless_overridden(variant.container, extra_output_args)
    cmd += extra_output_args
    cmd += [str(tmp_out)]
    return cmd


def variant_for_video_task(profile: EncodeProfile, variant: OutputVariant, task: VideoStreamTask) -> OutputVariant:
    cloned = OutputVariant.from_dict(output_variant_to_dict(variant))
    settings = task.settings
    cloned.backend = settings.backend or variant.backend
    cloned.ffmpeg_encoder = settings.ffmpeg_encoder or variant.ffmpeg_encoder
    cloned.resource_ids = list(settings.resource_ids or variant.resource_ids)
    cloned.height = settings.height
    cloned.rate_mode = settings.rate_mode or variant.rate_mode
    cloned.cq_value = settings.cq_value
    cloned.bitrate = settings.bitrate or variant.bitrate
    cloned.maxrate = settings.maxrate or variant.maxrate
    cloned.bufsize = settings.bufsize or variant.bufsize
    cloned.pix_fmt = settings.pix_fmt or variant.pix_fmt
    cloned.scale_flags = settings.scale_flags or variant.scale_flags
    if cloned.backend == BACKEND_CPU:
        cloned.cpu_codec = cloned.ffmpeg_encoder or profile.cpu_codec
        cloned.cpu_preset = settings.preset or variant.cpu_preset or profile.cpu_preset
        cloned.cpu_tune = settings.tune or variant.cpu_tune or profile.cpu_tune
    else:
        cloned.codec = cloned.ffmpeg_encoder or profile.codec
        cloned.preset = settings.preset or variant.preset or profile.preset
        cloned.tune = settings.tune or variant.tune or profile.tune
    extra_parts = [part for part in (variant.extra_video_args, settings.extra_args) if str(part).strip()]
    cloned.extra_video_args = " ".join(extra_parts)
    return cloned


def build_video_stream_command(
    ffmpeg_path: Path,
    src: Path,
    tmp_out: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
    task: VideoStreamTask,
    start_seconds: Optional[float] = None,
    duration_seconds: Optional[float] = None,
    resource_id: str = "",
) -> List[str]:
    task_variant = variant_for_video_task(profile, variant, task)
    return build_ffmpeg_command(
        ffmpeg_path,
        src,
        tmp_out,
        profile,
        task_variant,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        resource_id=resource_id,
        input_stream_index=task.input_stream_index,
        omit_video_encoder=task.use_muxer_default,
        copy_video_stream=task.settings.action in {"copy", "auto"} and not task.use_muxer_default,
        pix_fmt_override=task.settings.pix_fmt,
    )


def build_concat_command(
    ffmpeg_path: Path,
    list_file: Path,
    tmp_out: Path,
    profile: Optional[EncodeProfile] = None,
    variant: Optional[OutputVariant] = None,
) -> List[str]:
    extra_args = parse_ffmpeg_args(
        variant_setting(profile, variant, "extra_concat_args", "") if profile and variant else ""
    )
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
    cmd += faststart_args_unless_overridden(tmp_out.suffix, extra_args)
    cmd += extra_args
    cmd += [str(tmp_out)]
    return cmd


def build_audio_command(
    ffmpeg_path: Path,
    src: Path,
    tmp_audio: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
) -> List[str]:
    extra_input_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_input_args", ""))
    extra_audio_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_audio_args", ""))
    audio_codec = variant_audio_codec(profile, variant)
    audio_bitrate = str(variant_setting(profile, variant, "audio_bitrate", "") or "").strip()
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-stats_period",
        "1",
    ]
    cmd += extra_input_args
    cmd += [
        "-i",
        str(src),
        "-y",
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        audio_codec,
    ]
    if audio_codec.lower() != "copy" and audio_bitrate:
        cmd += ["-b:a", audio_bitrate]
    cmd += extra_audio_args
    cmd += [str(tmp_audio)]
    return cmd


def build_mux_command(
    ffmpeg_path: Path,
    video_in: Path,
    audio_in: Path,
    tmp_out: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
) -> List[str]:
    extra_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_mux_args", ""))
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-i",
        str(video_in),
        "-i",
        str(audio_in),
        "-y",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        "-shortest",
    ]
    cmd += faststart_args_unless_overridden(tmp_out.suffix, extra_args)
    cmd += extra_args
    cmd += [str(tmp_out)]
    return cmd


def ffmpeg_stream_letter(codec_type: str) -> str:
    return {
        "video": "v",
        "audio": "a",
        "subtitle": "s",
        "attachment": "t",
        "data": "d",
    }.get(str(codec_type).lower(), str(codec_type)[:1].lower() or "d")


def append_stream_attributes(
    cmd: List[str],
    codec_type: str,
    output_ordinal: int,
    tags: Dict[str, str],
    disposition: Dict[str, bool],
) -> None:
    letter = ffmpeg_stream_letter(codec_type)
    for key, raw_value in tags.items():
        if key in VOLATILE_STREAM_METADATA_TAGS:
            continue
        value = str(raw_value).strip()
        if value:
            cmd += [f"-metadata:s:{letter}:{output_ordinal}", f"{key}={value}"]
    active = [name for name, enabled in disposition.items() if enabled]
    cmd += [f"-disposition:{letter}:{output_ordinal}", "+".join(active) if active else "0"]


def build_stream_mux_command(
    ffmpeg_path: Path,
    video_inputs: List[Path],
    src: Path,
    tmp_out: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
    plan: ResolvedStreamPlan,
    sample_duration: Optional[float] = None,
) -> List[str]:
    cmd: List[str] = [str(ffmpeg_path), "-hide_banner"]
    for index, video_input in enumerate(video_inputs):
        if index < len(plan.video_tasks) and plan.video_tasks[index].start_time > 0:
            cmd += ["-itsoffset", format_seconds(plan.video_tasks[index].start_time)]
        cmd += ["-i", str(video_input)]
    source_input_index = len(video_inputs)
    cmd += ["-i", str(src), "-y"]

    for index, task in enumerate(plan.video_tasks):
        cmd += ["-map", f"{index}:v:0", f"-c:v:{task.output_ordinal}", "copy"]

    for stream in plan.mux_streams:
        letter = ffmpeg_stream_letter(stream.codec_type)
        cmd += ["-map", f"{source_input_index}:{stream.input_stream_index}"]
        codec_option = f"-c:{letter}:{stream.output_ordinal}"
        if stream.action == "copy" or stream.codec.lower() == "copy":
            cmd += [codec_option, "copy"]
        elif stream.codec:
            cmd += [codec_option, stream.codec]
        if stream.codec_type == "audio" and stream.action != "copy" and stream.codec.lower() != "copy":
            audio_bitrate = stream.bitrate or str(variant_setting(profile, variant, "audio_bitrate", "") or "").strip()
            if audio_bitrate:
                cmd += [f"-b:a:{stream.output_ordinal}", audio_bitrate]
        cmd += scope_stream_extra_args(stream.extra_args, stream.codec_type, stream.output_ordinal)

    if plan.preserve_metadata:
        cmd += ["-map_metadata", str(source_input_index)]
    else:
        cmd += ["-map_metadata", "-1"]
    if plan.preserve_chapters:
        cmd += ["-map_chapters", str(source_input_index)]
    else:
        cmd += ["-map_chapters", "-1"]
    for task in plan.video_tasks:
        append_stream_attributes(cmd, "video", task.output_ordinal, task.tags, task.disposition)
    for stream in plan.mux_streams:
        append_stream_attributes(cmd, stream.codec_type, stream.output_ordinal, stream.tags, stream.disposition)
    if sample_duration is not None and sample_duration > 0:
        cmd += ["-t", format_seconds(sample_duration)]
    extra_args = parse_ffmpeg_args(variant_setting(profile, variant, "extra_mux_args", ""))
    cmd += faststart_args_unless_overridden(tmp_out.suffix, extra_args)
    cmd += extra_args
    cmd += [str(tmp_out)]
    return cmd


def build_aux_stream_sample_command(
    ffmpeg_path: Path,
    src: Path,
    tmp_out: Path,
    stream: ResolvedMuxStream,
    use_muxer_default: bool = False,
    start_seconds: Optional[float] = None,
) -> List[str]:
    cmd = [str(ffmpeg_path), "-hide_banner"]
    if start_seconds is not None and start_seconds > 0:
        cmd += ["-ss", format_seconds(max(0.0, start_seconds - 0.1))]
    cmd += ["-i", str(src), "-y", "-map", f"0:{stream.input_stream_index}"]
    letter = ffmpeg_stream_letter(stream.codec_type)
    if not use_muxer_default:
        if stream.action == "copy" or stream.codec.lower() == "copy":
            cmd += [f"-c:{letter}", "copy"]
        elif stream.codec:
            cmd += [f"-c:{letter}", stream.codec]
    if stream.codec_type == "audio" and stream.bitrate and (use_muxer_default or stream.codec.lower() != "copy"):
        cmd += ["-b:a", stream.bitrate]
    cmd += scope_stream_extra_args(stream.extra_args, stream.codec_type, 0)
    if stream.codec_type == "video":
        cmd += ["-frames:v", "1"]
    elif stream.codec_type != "attachment":
        cmd += ["-t", "5.000"]
    append_stream_attributes(cmd, stream.codec_type, 0, stream.tags, stream.disposition)
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


def format_eta_duration(seconds: float) -> str:
    """Format a remaining-time estimate (in seconds) as a Japanese label."""
    if not math.isfinite(seconds) or seconds < 0:
        return ""
    total = int(round(seconds))
    if total <= 0:
        return "0秒"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}時間{minutes}分"
    if minutes > 0:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def estimate_remaining_seconds(elapsed: float, progress_delta: float, value: float) -> Optional[float]:
    """Estimate remaining seconds from elapsed time and overall progress (0-100).

    Returns None when there is not enough signal yet (too little elapsed time or
    no forward progress), so callers can show a "calculating" placeholder.
    """
    if elapsed < 1.0 or progress_delta <= 0.0 or value >= 100.0:
        return None
    rate = progress_delta / elapsed
    if rate <= 0.0:
        return None
    return (100.0 - value) / rate


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
            **no_window_subprocess_kwargs(),
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


def probe_media(ffprobe_path: Path, src: Path) -> MediaProbe:
    if not ffprobe_path.exists():
        raise RuntimeError(f"FFprobe unavailable: {ffprobe_path}")
    try:
        result = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-show_chapters",
                "-of",
                "json",
                str(src),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            **no_window_subprocess_kwargs(),
        )
    except Exception as exc:
        raise RuntimeError(f"ffprobe media probe failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"ffprobe media probe failed: exit {result.returncode}{suffix}")
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe media probe returned invalid JSON: {exc}") from exc
    probe = MediaProbe.from_ffprobe(src, data)
    if not probe.streams:
        raise RuntimeError("ffprobe did not report any streams")
    return probe


def probe_stream_first_packet_time(
    ffprobe_path: Path,
    src: Path,
    codec_type: str,
    ordinal: int,
) -> Optional[float]:
    letter = ffmpeg_stream_letter(codec_type)
    if codec_type == "attachment":
        return 0.0
    try:
        result = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-select_streams",
                f"{letter}:{max(0, int(ordinal))}",
                "-show_packets",
                "-show_entries",
                "packet=pts_time",
                "-read_intervals",
                "%+#1",
                "-of",
                "json",
                str(src),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            **no_window_subprocess_kwargs(),
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    packets = data.get("packets", []) if isinstance(data, dict) else []
    if not packets or not isinstance(packets[0], dict):
        return None
    return normalize_optional_float(packets[0].get("pts_time"), minimum=0.0)


def validate_output_against_plan(
    ffprobe_path: Path,
    output_path: Path,
    plan: ResolvedStreamPlan,
) -> Tuple[List[str], List[str]]:
    actual = probe_media(ffprobe_path, output_path)
    errors: List[str] = []
    warnings: List[str] = []
    expected_by_type: Dict[
        str,
        List[Tuple[str, Optional[int], Optional[int], Dict[str, str], Dict[str, bool]]],
    ] = {kind: [] for kind in STREAM_KINDS}
    for task in plan.video_tasks:
        expected_by_type["video"].append(
            (task.expected_codec, task.expected_width, task.expected_height, task.tags, task.disposition)
        )
    for stream in plan.mux_streams:
        expected_by_type.setdefault(stream.codec_type, []).append(
            (
                stream.expected_codec,
                stream.expected_width,
                stream.expected_height,
                stream.tags,
                stream.disposition,
            )
        )

    for codec_type, expected_items in expected_by_type.items():
        actual_items = [stream for stream in actual.streams if stream.codec_type == codec_type]
        if len(actual_items) != len(expected_items):
            errors.append(f"{codec_type} stream数: expected={len(expected_items)}, actual={len(actual_items)}")
            continue
        for ordinal, (expected_codec, expected_width, expected_height, tags, disposition) in enumerate(expected_items):
            stream = actual_items[ordinal]
            if expected_codec and stream.codec_name != expected_codec:
                errors.append(
                    f"{codec_type}:{ordinal} codec: expected={expected_codec}, actual={stream.codec_name or '-'}"
                )
            if expected_width is not None and stream.width != expected_width:
                errors.append(f"{codec_type}:{ordinal} width: expected={expected_width}, actual={stream.width or '-'}")
            if expected_height is not None and stream.height != expected_height:
                errors.append(
                    f"{codec_type}:{ordinal} height: expected={expected_height}, actual={stream.height or '-'}"
                )
            for key in ("language", "title"):
                expected_value = str(tags.get(key, "")).strip()
                if expected_value and stream.tags.get(key, "") != expected_value:
                    errors.append(
                        f"{codec_type}:{ordinal} {key}: expected={expected_value}, actual={stream.tags.get(key, '-') or '-'}"
                    )
            for key, expected_value in tags.items():
                if key in {"language", "title"} or key in VOLATILE_STREAM_METADATA_TAGS:
                    continue
                if expected_value and stream.tags.get(key) != expected_value:
                    warnings.append(f"{codec_type}:{ordinal} metadata {key} は保持されませんでした")
            expected_dispositions = {name for name, enabled in disposition.items() if enabled}
            actual_dispositions = {name for name, enabled in stream.disposition.items() if enabled}
            if actual_dispositions != expected_dispositions:
                errors.append(
                    f"{codec_type}:{ordinal} disposition: "
                    f"expected={','.join(sorted(expected_dispositions)) or '-'}, "
                    f"actual={','.join(sorted(actual_dispositions)) or '-'}"
                )

    if plan.preserve_chapters:
        if len(actual.chapters) != len(plan.chapters):
            errors.append(f"chapter数: expected={len(plan.chapters)}, actual={len(actual.chapters)}")
        else:
            for index, expected in enumerate(plan.chapters):
                actual_chapter = actual.chapters[index]
                if abs(actual_chapter.start_time - expected.start_time) > 0.05:
                    errors.append(f"chapter:{index} start_timeが保持されていません")
                if abs(actual_chapter.end_time - expected.end_time) > 0.05:
                    errors.append(f"chapter:{index} end_timeが保持されていません")
                title = expected.tags.get("title", "")
                if title and actual_chapter.tags.get("title", "") != title:
                    errors.append(f"chapter:{index} titleが保持されていません")
    if plan.preserve_metadata:
        for key, value in plan.source_tags.items():
            if key in {"encoder", "duration", "compatible_brands", "major_brand", "minor_version"}:
                continue
            if value and actual.tags.get(key) != value:
                warnings.append(f"metadata {key} は出力コンテナで保持されませんでした")
    return errors, warnings


def probe_has_audio(ffprobe_path: Path, src: Path) -> bool:
    if not ffprobe_path.exists():
        return False
    try:
        result = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(src),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            **no_window_subprocess_kwargs(),
        )
    except Exception as exc:
        raise RuntimeError(f"ffprobe audio probe failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"ffprobe audio probe failed: exit {result.returncode}{suffix}")
    return bool(result.stdout.strip())


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
        "variant_filename_template": variant.filename_template,
        "variant_input_dir": str(variant_input_dir(profile, variant)),
        "variant_output_dir": str(variant_output_dir(profile, variant)),
        "segment_minutes": variant_segment_minutes(profile, variant),
        "backend": variant.backend,
        "ffmpeg_encoder": encoder_codec(profile, variant),
        "resource_ids": variant_resource_ids(profile, variant),
        "split_encode_mode": variant.split_encode_mode,
        "use_gpu": variant_use_gpu(profile, variant),
        "gpu_index": variant_gpu_index(profile, variant),
        "codec": encoder_codec(profile, variant),
        "cpu_codec": variant_setting(profile, variant, "cpu_codec", profile.cpu_codec),
        "preset": variant_setting(profile, variant, "preset", profile.preset),
        "cpu_preset": variant_setting(profile, variant, "cpu_preset", profile.cpu_preset),
        "cpu_tune": variant_setting(profile, variant, "cpu_tune", profile.cpu_tune),
        "tune": variant_setting(profile, variant, "tune", profile.tune),
        "rate_mode": variant_setting(profile, variant, "rate_mode", profile.rate_mode),
        "cq_value": variant_setting(profile, variant, "cq_value", profile.cq_value),
        "bitrate": variant_setting(profile, variant, "bitrate", profile.bitrate),
        "maxrate": variant_setting(profile, variant, "maxrate", profile.maxrate),
        "bufsize": variant_setting(profile, variant, "bufsize", profile.bufsize),
        "pix_fmt": output_pix_fmt(profile, variant),
        "scale_flags": variant_setting(profile, variant, "scale_flags", profile.scale_flags),
        "audio_codec": variant_audio_codec(profile, variant),
        "audio_bitrate": variant_setting(profile, variant, "audio_bitrate", ""),
        "audio_container": variant_audio_container(profile, variant),
        "extra_input_args": variant_setting(profile, variant, "extra_input_args", ""),
        "extra_video_args": variant_setting(profile, variant, "extra_video_args", ""),
        "extra_audio_args": variant_setting(profile, variant, "extra_audio_args", ""),
        "extra_output_args": variant_setting(profile, variant, "extra_output_args", ""),
        "extra_concat_args": variant_setting(profile, variant, "extra_concat_args", ""),
        "extra_mux_args": variant_setting(profile, variant, "extra_mux_args", ""),
        "stream_rules": [asdict(rule) for rule in variant.stream_rules],
        "preserve_metadata": variant.preserve_metadata,
        "preserve_chapters": variant.preserve_chapters,
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


def stream_segment_dir_for(
    paths: AppPaths,
    src: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
    task: VideoStreamTask,
    plan: ResolvedStreamPlan,
) -> Path:
    return segment_dir_for(paths, src, profile, variant) / (f"video-{task.input_stream_index:03d}-{plan.fingerprint()}")


def segment_file_name(
    variant: OutputVariant,
    index: int,
    partial: bool = False,
    src: Optional[Path] = None,
) -> str:
    container = normalize_container_extension(variant.container)
    marker = ".partial" if partial else ""
    if src is not None:
        stem = safe_file_stem(src.stem)
        if len(stem) > SEGMENT_SOURCE_STEM_LIMIT:
            stem = stem[:SEGMENT_SOURCE_STEM_LIMIT].rstrip(".- ") or "video"
        return f"{stem}-{index + 1:03d}{marker}.{container}"
    return f"segment-{index:05d}{marker}.{container}"


def joined_video_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant) -> Path:
    container = normalize_container_extension(variant.container)
    return paths.tmp_dir / f"{job_key(src, profile, variant)}.video.{container}"


def stream_joined_video_path_for(
    paths: AppPaths,
    src: Path,
    profile: EncodeProfile,
    variant: OutputVariant,
    task: VideoStreamTask,
    plan: ResolvedStreamPlan,
) -> Path:
    container = normalize_container_extension(variant.container)
    return paths.tmp_dir / (
        f"{job_key(src, profile, variant)}.video-{task.input_stream_index:03d}-{plan.fingerprint()}.{container}"
    )


def temp_audio_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant) -> Path:
    container = variant_audio_container(profile, variant)
    return paths.tmp_dir / f"{job_key(src, profile, variant)}.audio.{container}"


def temp_output_path_for(paths: AppPaths, src: Path, profile: EncodeProfile, variant: OutputVariant) -> Path:
    container = normalize_container_extension(variant.container)
    return paths.tmp_dir / f"{job_key(src, profile, variant)}.final.{container}"


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
        "version": 4,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile_to_dict(profile),
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
                assigned_resource_id=str(item.get("assigned_resource_id", "")),
                assigned_slot=normalize_int(item.get("assigned_slot", 0), minimum=0, default=0),
                resolved_stream_plan=ResolvedStreamPlan.from_dict(item.get("resolved_stream_plan")),
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
