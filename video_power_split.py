from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# === Parameters (edit to your needs) ===
INPUT_DIR: Path = Path("1218_004wt_2")  # folder containing the videos
OUTPUT_ROOT: Path = Path("1218_004wt_2")  # where to place power folders and frames
VIDEO_EXTENSIONS: Sequence[str] = (".mp4", ".mov", ".mkv")
START_POWER_MW: int = 100  # first file gets this power
STEP_POWER_MW: int = 100     # increment per file (in modified-time order)
MOVE_FILES: bool = True     # True: move into power folder; False: copy
FFMPEG_CMD: str = "ffmpeg"  # path to ffmpeg.exe if not on PATH
HWACCEL: str = "auto"       # ffmpeg -hwaccel value (e.g., "auto", "cuda", "d3d11va")
HWACCEL_OUTPUT_FORMAT: str | None = None  # e.g., "cuda"; leave None if unsure
FRAME_RATE: str | None = None  # e.g., "30" to force fps; None to keep source
OUTPUT_IMAGE_EXT: str = "png"  # png or jpg
KEEP_EXISTING_FRAMES: bool = False  # False will overwrite frames folder if exists
DRY_RUN: bool = False  # True: only print actions, do not move or run ffmpeg
PRESERVE_INPUT_FPS: bool = True  # True: forbid fps override to avoid frame drop/dup
# ================================


def list_videos(folder: Path, exts: Iterable[str]) -> List[Path]:
    ext_set = {e.lower() for e in exts}
    videos = [p for p in folder.iterdir() if p.suffix.lower() in ext_set and p.is_file()]
    videos.sort(key=lambda p: p.stat().st_mtime)
    if not videos:
        raise FileNotFoundError(f"No videos with extensions {sorted(ext_set)} in {folder}")
    return videos


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_or_copy(src: Path, dst: Path, move: bool) -> None:
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def build_ffmpeg_cmd(ffmpeg: str, input_path: Path, output_pattern: Path) -> List[str]:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if HWACCEL:
        cmd += ["-hwaccel", HWACCEL]
        if HWACCEL_OUTPUT_FORMAT:
            cmd += ["-hwaccel_output_format", HWACCEL_OUTPUT_FORMAT]
    cmd += ["-i", str(input_path)]
    # Avoid frame drop/dup: use passthrough fps unless explicitly overridden and allowed.
    cmd += ["-fps_mode", "passthrough"]
    if FRAME_RATE and not PRESERVE_INPUT_FPS:
        cmd += ["-r", FRAME_RATE]
    cmd += ["-vsync", "0", "-frame_pts", "1", str(output_pattern)]
    return cmd


def extract_frames(ffmpeg: str, video_path: Path, frames_dir: Path) -> None:
    ensure_dir(frames_dir)
    output_pattern = frames_dir / f"%06d.{OUTPUT_IMAGE_EXT}"
    if not KEEP_EXISTING_FRAMES and frames_dir.exists():
        for f in frames_dir.iterdir():
            if f.is_file():
                f.unlink()
    cmd = build_ffmpeg_cmd(ffmpeg, video_path, output_pattern)
    print("FFmpeg cmd:", " ".join(cmd))
    if DRY_RUN:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {INPUT_DIR}")

    videos = list_videos(INPUT_DIR, VIDEO_EXTENSIONS)

    ffmpeg = FFMPEG_CMD

    current_power = START_POWER_MW
    rows: List[Tuple[str, str, str]] = []  # (video_name, power_folder, status)

    for video in videos:
        power_label = f"{current_power}mW"
        power_dir = OUTPUT_ROOT / power_label
        frames_dir = power_dir / "frames"

        ensure_dir(power_dir)

        target_video = power_dir / video.name
        print(f"Assign {video.name} -> {power_label}")
        if DRY_RUN:
            rows.append((video.name, power_label, "planned"))
        else:
            move_or_copy(video, target_video, MOVE_FILES)
            try:
                extract_frames(ffmpeg, target_video, frames_dir)
                rows.append((video.name, power_label, "ok"))
            except subprocess.CalledProcessError as e:
                rows.append((video.name, power_label, f"ffmpeg failed: {e}"))

        current_power += STEP_POWER_MW

    print("Done.")
    for name, power, status in rows:
        print(f"{name} => {power} : {status}")


if __name__ == "__main__":
    main()
