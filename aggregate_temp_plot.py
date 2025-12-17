from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

# === Parameters (edit as needed) ===
BASE_DIR: Path = Path("1217wang2")  # folder containing power subfolders like 100mW, 150mW, ...
FRAMES_CSV_NAME: str = "frames.csv"   # csv produced by roi_temperature_ocr.py
OUTPUT_COMBINED_CSV: Path = BASE_DIR / "combined_temp_vs_time.csv"
OUTPUT_FIG: Path = BASE_DIR / "temp_vs_time.png"
FPS: float | None = 8.77  # if set (e.g., 30.0), time axis is seconds; if None, use frame index
TEMP_COL_PREFIX: str = "temp"  # columns starting with this prefix are used (temp1, temp2, ...)
AGG_METHOD: str = "max"  # how to aggregate multiple temp columns: "mean" or "median"
DENOISE_ENABLED: bool = True
DENOISE_WINDOW: int = 31  # rolling median window (odd preferred)
DENOISE_THRESHOLD: float = 3.0  # replace points that deviate > threshold from rolling median
DOWNSAMPLE_EVERY: int | None = None  # e.g., 5 keeps every 5th point; None keeps all
FIG_SIZE: Sequence[float] = (10, 6)
ALPHA: float = 0.9
LINE_WIDTH: float = 1.3
MARKER_EVERY: int | None = None  # e.g., 50 to place markers every 50 points; None for no markers

# Series naming in the plot legend:
# - If True: use the parent folder name of frames.csv (e.g., "100mW")
# - If False: use parsed numeric power (e.g., "100 mW")
LABEL_FROM_PARENT_FOLDER: bool = True
# ===================================


def find_power_folders(base: Path, csv_name: str) -> List[tuple[int, Path]]:
    powers: List[tuple[int, Path]] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        csv_path = p / csv_name
        if not csv_path.exists():
            continue
        # Keep a stable ordering; do not require any naming convention (e.g. "100mW").
        powers.append((0, p))
    powers.sort(key=lambda x: x[1].name.lower())
    if not powers:
        raise FileNotFoundError(f"No subfolders with {csv_name} under {base}")
    return powers


def load_and_prepare(
    csv_path: Path,
    fps: float | None,
    agg_method: str,
    temp_prefix: str,
    downsample: int | None,
    denoise_enabled: bool,
    denoise_window: int,
    denoise_threshold: float,
):
    df = pd.read_csv(csv_path)
    temp_cols = [c for c in df.columns if c.startswith(temp_prefix)]
    if not temp_cols:
        raise ValueError(f"No columns starting with '{temp_prefix}' in {csv_path}")
    # ensure numeric
    for c in temp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if agg_method == "mean":
        df["temp_agg"] = df[temp_cols].mean(axis=1, skipna=True)
    elif agg_method == "median":
        df["temp_agg"] = df[temp_cols].median(axis=1, skipna=True)
    elif agg_method == "max":
        df["temp_agg"] = df[temp_cols].max(axis=1, skipna=True)
    else:
        raise ValueError("AGG_METHOD must be 'mean' or 'median'")

    # Denoise with rolling median; replace spikes that deviate more than threshold.
    if denoise_enabled:
        med = df["temp_agg"].rolling(denoise_window, center=True, min_periods=1).median()
        mask = (df["temp_agg"] - med).abs() > denoise_threshold
        df.loc[mask, "temp_agg"] = med[mask]

    # derive time axis from frame index (sorted by image name to be safe)
    df = df.sort_values("image")
    frame_idx = range(len(df))
    if fps:
        df["time"] = [i / fps for i in frame_idx]
    else:
        df["time"] = list(frame_idx)

    if downsample and downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)

    return df[["time", "temp_agg"]]


def main() -> None:
    base = BASE_DIR
    if not base.exists():
        raise FileNotFoundError(f"BASE_DIR does not exist: {base}")

    power_folders = find_power_folders(base, FRAMES_CSV_NAME)

    combined_rows: List[Dict[str, float | int | str]] = []

    plt.figure(figsize=FIG_SIZE)

    for power_mw, folder in power_folders:
        csv_path = folder / FRAMES_CSV_NAME
        df = load_and_prepare(
            csv_path,
            FPS,
            AGG_METHOD,
            TEMP_COL_PREFIX,
            DOWNSAMPLE_EVERY,
            DENOISE_ENABLED,
            DENOISE_WINDOW,
            DENOISE_THRESHOLD,
        )

        for _, row in df.iterrows():
            combined_rows.append(
                {
                    "series": folder.name,
                    "time": row["time"],
                    "temp": row["temp_agg"],
                }
            )

        label = folder.name if LABEL_FROM_PARENT_FOLDER else folder.name
        marker_opts = {"markevery": MARKER_EVERY} if MARKER_EVERY else {}
        plt.plot(df["time"], df["temp_agg"], label=label, alpha=ALPHA, linewidth=LINE_WIDTH, **marker_opts)

    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(OUTPUT_COMBINED_CSV, index=False)

    plt.xlabel("Time (s)" if FPS else "Frame index")
    plt.ylabel("Temperature (°C)")
    plt.title("Temperature vs Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=200)

    print(f"Saved combined CSV: {OUTPUT_COMBINED_CSV}")
    print(f"Saved plot: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
