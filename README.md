# NvencArchiveGui

Windows native GUI encoder for OBS recording archive jobs.

## Features

- Scan `OBSData`
- Select multiple videos
- Create 4K output and MP4 output
- Run up to two FFmpeg/NVENC jobs in parallel
- Start / Pause / Resume / Stop
- Show live logs
- Save job logs to `tmp/logs`
- Save resume state to `tmp/encoder_state.json`
- Skip already finished outputs
- Move source files to `SourceData` only when requested outputs are complete
- Download FFmpeg automatically when `ffmpeg\ffmpeg.exe` is missing

## Folder layout

Place the exe in your work folder.

```text
work-folder
├─ NvencArchiveGui.exe
├─ ffmpeg
│  └─ ffmpeg.exe
├─ OBSData
├─ SourceData
├─ SourceData(MP4)
├─ SourceData(4K)
└─ tmp
```

If `ffmpeg\ffmpeg.exe` does not exist, the app can download FFmpeg on first run.

## Run from source

```powershell
py -m pip install -r requirements-dev.txt
py -m ffmpeg_nvenc_gui
```

or:

```powershell
py src\ffmpeg_nvenc_gui\app.py
```

## Build exe locally

```powershell
py -m pip install -r requirements-dev.txt
py -m PyInstaller --noconfirm --onefile --windowed --name NvencArchiveGui --paths src src\ffmpeg_nvenc_gui\app.py
```

The exe will be created at:

```text
dist\NvencArchiveGui.exe
```

## Resume behavior

This app does not resume an incomplete MP4 from the middle.
It safely restarts only unfinished jobs.
Already completed outputs are skipped.

## CI/CD

- Push to `develop`: tests only
- Push to `main`: tests + Windows exe build + artifact upload
- Pull request to `main` or `develop`: tests only
