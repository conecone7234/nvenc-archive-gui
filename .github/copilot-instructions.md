# Copilot Review Instructions

リポジトリ全体のエージェント向け恒久指示は `AGENTS.md` を正本とする。
レビュー挙動は、このファイルと `AGENTS.md` の両方に沿わせること。

This repository is a Windows Tkinter app for FFmpeg archive encoding. Review
changes with the behavior of `ffmpeg_nvenc_gui.app` and
`ffmpeg_nvenc_gui.core` in mind, not as a generic web or CLI project.

## Response Language

- Pull request review overviews, inline review comments, requested-change
  summaries, and PR conversation comments must be written in Japanese.
- Do not write explanatory review prose in English. Translate headings,
  summaries, severity explanations, and action requests into Japanese.
- Keep technical identifiers, commands, option names, and error strings
  unchanged when quoting code or logs.

## Project Priorities

- Keep the main screen focused on starting work, pause/stop controls, logs, and
  file progress. Put detailed codec, output, and folder controls in profile or
  settings flows.
- Preserve the profile-driven workflow. Input, output, archive, temp, and output
  variant behavior should come from `EncodeProfile` and `OutputVariant`, not
  hard-coded folder names.
- Preserve segment-based resume behavior. Do not pause or suspend an active
  encoder process; completed segments should remain resumable after stop,
  failure, or restart.
- Treat CPU and GPU paths as first-class. NVENC, QSV, AMF, and CPU behavior
  should keep clear user-facing labels, compatible encoders, and sensible
  fallback behavior.
- Keep Windows paths, filenames with spaces, and non-ASCII source names safe.
  Build FFmpeg commands as argument lists instead of shell-joined strings.

## Review Focus

- Validate both save-time and run-time profile behavior. A profile with only
  disabled output variants should not be accepted as runnable.
- Keep output target collision checks in sync with filename templates,
  containers, folder names, and enabled output variants.
- Keep rate-control validation and widget state consistent. CQ/CRF is only
  valid where the selected backend supports it, and bitrate fields should match
  the selected rate mode.
- Fail fast on real FFprobe or FFmpeg errors, but preserve intentional fallback
  behavior when optional tooling is genuinely missing.
- Check small-window usability for Tkinter changes. Scrollable tabs, controls,
  labels, and validation messages should not overlap or become unreachable.
- Do not introduce broad rewrites or new frameworks for narrow fixes. Follow the
  existing dataclass, Tkinter, and test patterns unless the change requires a
  larger refactor.

## Validation Expectations

- Add or update tests for changes in profile normalization, command building,
  resume behavior, FFprobe/FFmpeg fallback paths, and validation rules.
- Run `python -m ruff check .`, `python -m ruff format --check .`, and
  `python -m pytest -q` before merging code changes.
- For release workflow changes, preserve the explicit branch channels:
  `main` for stable releases, `preview` for preview prereleases, and `develop`
  for nightly prereleases.
