# NvencArchiveGui

Windows native GUI encoder for profile-based FFmpeg/NVENC archive jobs.

## Features

- Home screen focused on Start / Pause / Stop, logs, and file-by-file progress
- Profile-based input, output, and source archive folders
- Multiple output variants per source file, such as `Master 2160p` and `Reference 1080p`
- Resolution presets plus custom output height
- CQ/CRF, VBR, ABR, and CBR modes, with irrelevant bitrate fields disabled in the UI
- Manual GPU selection with CPU-only fallback when no NVIDIA GPU is detected
- Parallel job count per profile
- Automatic FFmpeg and FFprobe install when the bundled executables are missing
- Segment-based fault tolerance: completed time slices are kept and unfinished work can be resumed
- Source files move to the profile archive folder only after all requested outputs are complete

## Folder Layout

The app no longer depends on fixed folders named `OBSData`, `SourceData(MP4)`, or `SourceData(4K)`.
Profiles define their own folders. A fresh profile defaults to:

```text
work-folder
|-- NvencArchiveGui.exe
|-- ffmpeg
|   |-- ffmpeg.exe
|   `-- ffprobe.exe
|-- Incoming
|-- Encoded
|   |-- master-2160p
|   `-- reference-1080p
|-- SourceArchive
|-- profiles.json
`-- tmp
    |-- encoder_state.json
    |-- logs
    `-- segments
```

Profile folders are created when the profile is executed, not at app startup.

## Resume Behavior

The app does not pause an active video encoder process in the middle of a write.
Instead, each output is encoded as fixed-duration segments. Pause waits until the
current segment finishes, and Stop leaves completed segments in `tmp/segments`.
Use `保存状態から再開` to continue after an app crash, forced shutdown, or manual stop.

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

- Pull request to `main`, `preview`, or `develop`: tests only
- Push or merge to `main`: tests + Windows exe build + stable GitHub Release
- Push or merge to `preview`: tests + Windows exe build + preview pre-release
- Push or merge to `develop`: tests + Windows exe build + nightly pre-release

Release channels:

- `main`: stable release, tag `vX.X.X`, marked as GitHub Latest
- `preview`: preview release, tag `vX.X.X-preview.<run>`, marked as pre-release
- `develop`: nightly release, tag `vX.X.X-nightly.<run>`, marked as pre-release

`pyproject.toml` is the source of the base version. Bump `project.version`
before creating another stable release from `main`; the workflow intentionally
fails if the stable tag already exists.
