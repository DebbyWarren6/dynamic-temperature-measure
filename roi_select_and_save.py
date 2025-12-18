from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2

# === Parameters ===
INPUT_DIR: Path = Path("1218_004wt/100mW/frames")  # folder containing images
ROI_CONFIG: Path = Path("roi_config.json")  # where to save ROI list as JSON
IMAGE_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp")
SORT_BY: str = "mtime"  # or "name"
# ===================


def collect_images(folder: Path, exts: Iterable[str], sort_by: str) -> List[Path]:
    normalized_exts = {ext.lower() for ext in exts}
    candidates = [p for p in folder.iterdir() if p.suffix.lower() in normalized_exts and p.is_file()]
    sort_by = sort_by.strip().lower()
    if sort_by == "name":
        images = sorted(candidates, key=lambda p: p.name.lower())
    elif sort_by == "mtime":
        images = sorted(candidates, key=lambda p: p.stat().st_mtime)
    else:
        raise ValueError("SORT_BY must be 'name' or 'mtime'")
    if not images:
        raise FileNotFoundError(f"No images with extensions {sorted(normalized_exts)} in {folder}")
    return images


def request_rois(image_path: Path) -> List[Tuple[int, int, int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read first image: {image_path}")

    window_name = "Select ROIs (drag boxes, press ENTER/SPACE to finish, ESC to cancel)"
    cv2.imshow(window_name, image)
    rois = cv2.selectROIs(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if rois is None or len(rois) == 0:
        raise RuntimeError("No ROI selected; please rerun and draw one or more boxes")

    roi_list: List[Tuple[int, int, int, int]] = []
    for roi in rois:
        x, y, w, h = roi
        if int(w) <= 0 or int(h) <= 0:
            continue
        roi_list.append((int(x), int(y), int(w), int(h)))

    if not roi_list:
        raise RuntimeError("No valid ROI selected; please rerun and draw one or more boxes")

    roi_list.sort(key=lambda r: r[1])
    return roi_list


def save_rois(rois: List[Tuple[int, int, int, int]], path: Path) -> None:
    payload = {"rois": rois}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(rois)} ROIs to {path}")


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {INPUT_DIR}")

    images = collect_images(INPUT_DIR, IMAGE_EXTENSIONS, SORT_BY)
    first_image = images[0]
    rois = request_rois(first_image)
    save_rois(rois, ROI_CONFIG)


if __name__ == "__main__":
    main()
