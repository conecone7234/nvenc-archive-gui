# NvencArchiveGui

Windows native GUI encoder for FFmpeg/NVENC archive jobs.

## Features

- Home screen focused on the Start/Pause/Resume cycle, Stop, logs, and file-by-file progress
- Settings-based default input, output, and source archive folders
- Multiple output profiles per source file, such as `UP Convert 4K` and `ReEncode Original Pixel`
- Per-output-profile input/output folder overrides, file name templates, segment length, resolution, container, codec, rate control, resources, audio, and advanced FFmpeg arguments
- Resolution presets plus custom output height
- CQ/CRF, VBR, ABR, and CBR modes
- Detected hardware resources by default, with advanced manual GPU resource registration when needed
- Segment parallelism follows the selected resource's slot count
- One-source-at-a-time processing order, with each source's output profiles completed before the next source starts
- FFprobe-based inventory and preflight for every video, audio, subtitle, attachment, data, chapter, and metadata stream
- Ordered stream rules per output profile plus persistent per-file overrides from the progress list
- Every normal video stream is encoded as an independently resumable child job, then all selected streams are muxed into one output
- Container compatibility fallback through the local FFmpeg muxer default, with confirmation before encoding
- Post-mux verification of stream counts, codecs, chapters, language/title metadata, and dispositions
- Automatic FFmpeg and FFprobe install when the bundled executables are missing
- Segment-based fault tolerance: completed time slices are kept and unfinished work can be resumed
- Source files move to the archive folder only after all requested outputs are complete

## Folder Layout

The app no longer depends on fixed folders named `OBSData`, `SourceData(MP4)`, or `SourceData(4K)`.
Settings define their own folders. A fresh setting defaults to:

```text
work-folder
|-- NvencArchiveGui.exe
|-- ffmpeg
|   |-- ffmpeg.exe
|   `-- ffprobe.exe
|-- Incoming
|-- output
|   `-- {source}
|       |-- {source}.mp4
|       |-- up-convert-4k
|       `-- reencode-original-pixel
|-- profiles.json
`-- tmp
    |-- encoder_state.json
    |-- stream_overrides.json
    |-- logs
    `-- segments
```

Folders are created when the setting is executed, not at app startup.

## Resume Behavior

The app does not pause an active video encoder process in the middle of a write.
Instead, each normal video stream in an output profile is encoded as fixed-duration video-only segments.
The segment length is configured per output profile, with the setting default used
when the output profile leaves it blank. Segments for one output profile can run
in parallel according to the selected resource's slot count. Each video stream
may use its own encoder, backend, and resource. After every video stream is
concatenated, the selected audio, subtitle, cover, attachment, and data streams
are muxed with chapters and metadata. The output is FFprobe-validated before it
replaces the final file. Pause waits before launching more segments, and the
primary run button changes to Resume while resumable state exists. Stop is the
explicit restart path: it clears the saved state and removes temporary segment
files for the current run.
Use `再開` to continue after an app crash or forced shutdown when saved state remains.

## Run From Source

```powershell
py -m pip install -r requirements-dev.txt
py -m ffmpeg_nvenc_gui
```

or:

```powershell
py src\ffmpeg_nvenc_gui\app.py
```

## Build Exe Locally

```powershell
py -m pip install -r requirements-dev.txt
py -m PyInstaller --noconfirm --onefile --windowed --name NvencArchiveGui --paths src src\ffmpeg_nvenc_gui\app.py
```

The exe will be created at:

```text
dist\NvencArchiveGui.exe
```

## CI/CD

- Pull request to `main`, `preview`, or `develop`: lint + tests only
- Push or merge to `main`: lint + tests + Windows exe build + stable GitHub Release
- Push or merge to `preview`: lint + tests + Windows exe build + preview pre-release
- Push or merge to `develop`: lint + tests + Windows exe build + nightly pre-release

Release channels:

- `main`: stable release, tag `vX.X.X`, marked as GitHub Latest
- `preview`: preview release, tag `vX.X.X-preview.<run>`, marked as pre-release
- `develop`: nightly release, tag `vX.X.X-nightly.<run>`, marked as pre-release

`pyproject.toml` is the source of the base version. Bump `project.version`
before creating another stable release from `main`; the workflow intentionally
fails if the stable tag already exists.
