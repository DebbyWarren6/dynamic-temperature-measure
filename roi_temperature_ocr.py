from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

# === Parameters (edit these) ===
# Process exactly one experiment folder per run.
# Change this to the experiment you want to process.
EXPERIMENT_DIR: Path = Path("1218_004wt_2")  # folder containing power subfolders like 100mW, 150mW, ...

# Power folder naming rule: e.g. 100mW, 150mW ...
POWER_DIR_REGEX: str = r"^(\d+)[mM][wW]$"
FRAMES_SUBDIR: str = "frames"
OUTPUT_CSV_NAME: str = "frames.csv"

# ROI config saved by roi_select_and_save.py (JSON with list of rois [[x,y,w,h], ...])
ROI_CONFIG: Path = Path("roi_config.json")
ROI_PADDING: int = 4

# Image extensions to include.
IMAGE_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp")
SORT_BY: str = "name"  # "name" or "mtime"

# OCR backends: "rapidocr" (default) or "paddle" (GPU-capable if installed)
BACKEND: str = "rapidocr"
MIN_CONFIDENCE: float = 0.5  # discard predictions below this

# Batch stitching (key speed-up): stitch many ROI crops into one mosaic, OCR once, then split back.
STITCH_ENABLED: bool = True
STITCH_BATCH_SIZE: int = 128  # number of frames (tiles) per OCR call, per ROI index
STITCH_MAX_COLS: int = 8     # mosaic columns; rows will expand as needed
STITCH_GAP_PX: int = 4       # gap between tiles in mosaic

# OCR validation / error handling
FAIL_FAST: bool = True
REQUIRE_TEMP_PER_TILE: bool = True  # if True and any tile yields no temperature -> error
FALLBACK_SINGLE_TILE_ON_FAILURE: bool = True  # re-OCR failed tiles individually before error

# Fallback-only rescue for occasional frames where RapidOCR finds no text at native tile size.
# This does NOT affect normal tiles; it runs only when a tile has no extracted temperature.
FALLBACK_ADAPTIVE_THRESHOLD: bool = True

# Simple timing output
TIMING_ENABLED: bool = True
TIMING_PRINT_EVERY_BATCH: int = 1  # set to N>1 to print every N batches
TIMING_DETAILED: bool = True  # print breakdown: imread/cache/crop/mosaic/ocr

# Concurrent prefetch: overlaps cv2.imread (IO+decode) with OCR.
PREFETCH_ENABLED: bool = True
PREFETCH_WORKERS: int = 8
EAGER_PREFETCH_ALL: bool = False  # if True, queue reading of all frames in this folder upfront

# PaddleOCR options (requires paddlepaddle-gpu + paddleocr installed and NVIDIA GPU).
PADDLE_LANG: str = "en"  # e.g., "ch" for Chinese
PADDLE_DEVICE: str = "gpu:0"
PADDLE_USE_TEXTLINE_ORIENTATION: bool = False
PADDLE_SHOW_LOG: bool = False

# RapidOCR options
RAPIDOCR_PARAMS: dict = {
    # Keep defaults; expose here for future overrides.
}
# ==============================


def collect_images(folder: Path, exts: Iterable[str]) -> List[Path]:
    normalized_exts = {ext.lower() for ext in exts}
    candidates = [p for p in folder.iterdir() if p.suffix.lower() in normalized_exts and p.is_file()]

    sort_by = SORT_BY.strip().lower()
    if sort_by == "name":
        images = sorted(candidates, key=lambda p: p.name.lower())
    elif sort_by == "mtime":
        images = sorted(candidates, key=lambda p: p.stat().st_mtime)
    else:
        raise ValueError(f"Unsupported SORT_BY: {SORT_BY!r} (use 'name' or 'mtime')")
    if not images:
        raise FileNotFoundError(f"No images with extensions {sorted(normalized_exts)} in {folder}")
    return images


def load_rois(config_path: Path) -> List[Tuple[int, int, int, int]]:
    if not config_path.exists():
        raise FileNotFoundError(f"ROI_CONFIG not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rois_raw = data.get("rois") if isinstance(data, dict) else data
    if not isinstance(rois_raw, list):
        raise ValueError("ROI config must be a list or dict with key 'rois'")

    roi_list: List[Tuple[int, int, int, int]] = []
    for roi in rois_raw:
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            continue
        x, y, w, h = roi
        if int(w) <= 0 or int(h) <= 0:
            continue
        roi_list.append((int(x), int(y), int(w), int(h)))

    if not roi_list:
        raise ValueError("ROI config contains no valid entries")

    # Sort by y (top to bottom) for consistent ordering.
    roi_list.sort(key=lambda r: r[1])
    return roi_list


def clamp_roi(x: int, y: int, w: int, h: int, width: int, height: int, pad: int) -> Tuple[int, int, int, int]:
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, width)
    y1 = min(y + h + pad, height)
    return x0, y0, x1 - x0, y1 - y0


def crop_from_image(image_bgr: np.ndarray, roi: Tuple[int, int, int, int], pad: int) -> np.ndarray:
    x, y, w, h = roi
    x, y, w, h = clamp_roi(x, y, w, h, image_bgr.shape[1], image_bgr.shape[0], pad)
    return image_bgr[y : y + h, x : x + w]


def extract_temperatures(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    temps: List[str] = []
    for line in lines:
        match = re.search(r"([-+]?\d+(?:\.\d+)?)[°]?", line)
        if match:
            temps.append(match.group(1))
    return temps


def _tile_to_raw_text(items: List[Tuple[float, float, str]]) -> str:
    items.sort(key=lambda x: x[1])
    texts = [s for _, __, s in items if s]
    return "\n".join(texts).strip()


def _adaptive_threshold_bgr(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def _ocr_tile_temperature_with_rescue(tile_bgr: np.ndarray, ocr_engine, backend: str) -> Tuple[str, str]:
    """Return (temp, raw_text) for a single tile.

    Normal attempt: OCR on tile as-is.
    Rescue attempts (only if needed): adaptive threshold.
    """

    def _attempt(img: np.ndarray) -> Tuple[str, str]:
        items = _ocr_items(img, ocr_engine, backend)
        raw = _tile_to_raw_text(items)
        temps = extract_temperatures(raw)
        return (temps[0] if temps else ""), raw

    temp, raw = _attempt(tile_bgr)
    if temp:
        return temp, raw

    if FALLBACK_ADAPTIVE_THRESHOLD:
        thr = _adaptive_threshold_bgr(tile_bgr)
        temp2, raw2 = _attempt(thr)
        if temp2:
            return temp2, raw2

    return "", raw


def _box_y_center(box: object) -> float:
    """Compute y-center from a RapidOCR box.

    Expected box format: 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    try:
        ys = [float(p[1]) for p in box]
        return sum(ys) / len(ys)
    except Exception:
        return 0.0


def _box_center(box: object) -> Tuple[float, float]:
    try:
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    except Exception:
        return (0.0, 0.0)


def write_csv(rows: List[Tuple[str, List[str], List[str]]], output_path: Path, roi_count: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["image"]
        header.extend([f"temp{i+1}" for i in range(roi_count)])
        header.extend([f"raw_text{i+1}" for i in range(roi_count)])
        writer.writerow(header)

        for image_name, temps_by_roi, raw_by_roi in rows:
            temps_padded = list(temps_by_roi) + [""] * (roi_count - len(temps_by_roi))
            raws_padded = list(raw_by_roi) + [""] * (roi_count - len(raw_by_roi))
            writer.writerow([image_name, *temps_padded[:roi_count], *raws_padded[:roi_count]])


def _init_ocr_engine(backend: str):
    backend = backend.lower().strip()
    if backend == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR(**RAPIDOCR_PARAMS)
    if backend == "paddle":
        from paddleocr import PaddleOCR

        return PaddleOCR(
            lang=PADDLE_LANG,
            use_textline_orientation=PADDLE_USE_TEXTLINE_ORIENTATION,
            device=PADDLE_DEVICE,
            show_log=PADDLE_SHOW_LOG,
        )
    raise ValueError(f"Unsupported BACKEND: {backend!r} (use 'rapidocr' or 'paddle')")


def _ocr_items(image_bgr: np.ndarray, ocr_engine, backend: str) -> List[Tuple[float, float, str]]:
    """Return list of (cx, cy, text) items with confidence filtered."""
    backend = backend.lower().strip()
    items: List[Tuple[float, float, str]] = []

    if backend == "rapidocr":
        result, _ = ocr_engine(image_bgr)
        if not result:
            return []
        for entry in result:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            box, text, score = entry[0], entry[1], entry[2]
            if not isinstance(text, str):
                continue
            try:
                score_f = float(score)
            except Exception:
                score_f = 1.0
            if score_f < MIN_CONFIDENCE:
                continue
            cx, cy = _box_center(box)
            items.append((cx, cy, text.strip()))
        return items

    # paddle
    result = ocr_engine.ocr(image_bgr, cls=False)
    if not result:
        return []
    for block in result:
        if not isinstance(block, (list, tuple)):
            continue
        for entry in block:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            box = entry[0]
            rec_part = entry[1]
            if not isinstance(rec_part, (list, tuple)) or len(rec_part) < 2:
                continue
            text, score = rec_part[0], rec_part[1]
            if not isinstance(text, str):
                continue
            try:
                score_f = float(score)
            except Exception:
                score_f = 1.0
            if score_f < MIN_CONFIDENCE:
                continue
            cx, cy = _box_center(box)
            items.append((cx, cy, text.strip()))
    return items


def _make_mosaic(tiles: List[np.ndarray], max_cols: int, gap: int) -> Tuple[np.ndarray, int, int, int, int]:
    """Return (mosaic, cols, rows, tile_w, tile_h). All tiles must be same size."""
    if not tiles:
        raise ValueError("No tiles to mosaic")
    tile_h, tile_w = tiles[0].shape[:2]
    for t in tiles:
        if t.shape[:2] != (tile_h, tile_w):
            raise ValueError("All tiles must have same size; check ROI consistency")

    cols = min(max_cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    mosaic_w = cols * tile_w + (cols - 1) * gap
    mosaic_h = rows * tile_h + (rows - 1) * gap
    canvas = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        x0 = c * (tile_w + gap)
        y0 = r * (tile_h + gap)
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
    return canvas, cols, rows, tile_w, tile_h


def _ocr_mosaic_tiles(
    tiles: List[np.ndarray],
    tile_names: List[str],
    ocr_engine,
    backend: str,
    batch_label: str,
    timing: Dict[str, float] | None = None,
) -> Tuple[List[str], List[str]]:
    """OCR a batch of ROI tiles via mosaic and return (temp_list, raw_text_list) aligned to tiles."""
    if not tiles:
        return [], []

    if len(tile_names) != len(tiles):
        raise ValueError("tile_names length must match tiles length")

    if not STITCH_ENABLED or len(tiles) == 1:
        temps_out: List[str] = []
        raws_out: List[str] = []
        if timing is not None:
            timing.setdefault("mosaic_s", 0.0)
            timing.setdefault("ocr_items_s", 0.0)
            timing.setdefault("fallback_s", 0.0)
        for t in tiles:
            t0 = time.perf_counter()
            temp, raw = _ocr_tile_temperature_with_rescue(t, ocr_engine, backend)
            t1 = time.perf_counter()
            if timing is not None:
                timing["fallback_s"] = timing.get("fallback_s", 0.0) + (t1 - t0)
            temps_out.append(temp)
            raws_out.append(raw)
        return temps_out, raws_out

    mosaic_t0 = time.perf_counter()
    mosaic, cols, _rows, tile_w, tile_h = _make_mosaic(tiles, STITCH_MAX_COLS, STITCH_GAP_PX)
    mosaic_t1 = time.perf_counter()
    ocr_items_t0 = time.perf_counter()
    items = _ocr_items(mosaic, ocr_engine, backend)
    ocr_items_t1 = time.perf_counter()

    if timing is not None:
        timing["mosaic_s"] = timing.get("mosaic_s", 0.0) + (mosaic_t1 - mosaic_t0)
        timing["ocr_items_s"] = timing.get("ocr_items_s", 0.0) + (ocr_items_t1 - ocr_items_t0)
        timing.setdefault("fallback_s", 0.0)

    texts_by_tile: List[List[Tuple[float, str]]] = [[] for _ in range(len(tiles))]

    for cx, cy, text in items:
        col = int(cx // (tile_w + STITCH_GAP_PX))
        row = int(cy // (tile_h + STITCH_GAP_PX))
        if col < 0 or row < 0 or col >= cols:
            continue
        idx = row * cols + col
        if 0 <= idx < len(tiles):
            texts_by_tile[idx].append((cy, text))

    temps_out: List[str] = []
    raws_out: List[str] = []
    failed_indices: List[int] = []

    for i, texts in enumerate(texts_by_tile):
        texts.sort(key=lambda x: x[0])
        raw = "\n".join([t for _, t in texts if t]).strip()
        temps = extract_temperatures(raw)
        temp = temps[0] if temps else ""

        if REQUIRE_TEMP_PER_TILE and not temp:
            failed_indices.append(i)
        temps_out.append(temp)
        raws_out.append(raw)

    # Optional fallback: run per-tile OCR for failed tiles to reduce false failures from mosaic assignment.
    if failed_indices and FALLBACK_SINGLE_TILE_ON_FAILURE:
        for i in failed_indices:
            t = tiles[i]
            fb_t0 = time.perf_counter()
            temp2, raw2 = _ocr_tile_temperature_with_rescue(t, ocr_engine, backend)
            fb_t1 = time.perf_counter()
            if timing is not None:
                timing["fallback_s"] = timing.get("fallback_s", 0.0) + (fb_t1 - fb_t0)
            if temp2:
                temps_out[i] = temp2
                raws_out[i] = raw2

        # recompute failures after fallback
        still_failed: List[int] = [i for i in failed_indices if not temps_out[i]]
        failed_indices = still_failed

    if failed_indices and FAIL_FAST:
        errors = [
            f"{batch_label} {tile_names[i]}: no temperature extracted; raw={raws_out[i]!r}"
            for i in failed_indices
        ]
        raise RuntimeError("OCR failed:\n" + "\n".join(errors[:20]))
    return temps_out, raws_out


def _find_frames_folders(experiment_dir: Path) -> List[Path]:
    """Find all subfolders containing a frames directory.

    This intentionally does NOT rely on folder naming like "100mW".
    """

    folders: List[Path] = []
    for p in experiment_dir.iterdir():
        if not p.is_dir():
            continue
        frames_dir = p / FRAMES_SUBDIR
        if frames_dir.exists() and frames_dir.is_dir():
            folders.append(p)
    folders.sort(key=lambda x: x.name.lower())
    return folders


def _process_frames_folder(frames_dir: Path, output_csv: Path, rois: List[Tuple[int, int, int, int]], ocr_engine, backend: str) -> None:
    images = collect_images(frames_dir, IMAGE_EXTENSIONS)
    roi_count = len(rois)

    folder_t0 = time.perf_counter()

    results: List[Tuple[str, List[str], List[str]]] = []

    # Process ROI-by-ROI in batches; this enables stitching ROI crops into mosaics.
    images_bgr: Dict[str, np.ndarray] = {}

    cache_lock = Lock()
    stats_lock = Lock()

    imread_s_total = 0.0
    imread_count = 0
    cache_hits = 0

    def _get_image(p: Path) -> np.ndarray:
        nonlocal imread_s_total, imread_count, cache_hits
        with cache_lock:
            cached = images_bgr.get(p.name)
            if cached is not None:
                cache_hits += 1
                return cached
        t0 = time.perf_counter()
        img = cv2.imread(str(p))
        t1 = time.perf_counter()
        with stats_lock:
            imread_s_total += t1 - t0
            imread_count += 1
        if img is None:
            raise RuntimeError(f"Cannot read image: {p}")
        with cache_lock:
            images_bgr[p.name] = img
        return img

    # Initialize per-image buffers
    temps_map: Dict[str, List[str]] = {p.name: [""] * roi_count for p in images}
    raw_map: Dict[str, List[str]] = {p.name: [""] * roi_count for p in images}

    executor: ThreadPoolExecutor | None = None
    if PREFETCH_ENABLED and PREFETCH_WORKERS > 0:
        executor = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS)

    try:
        pending: Dict[str, Future[None]] = {}
        pending_lock = Lock()

        def _submit_prefetch(batch_paths: List[Path]) -> None:
            if executor is None:
                return
            for pp in batch_paths:
                key = pp.name
                with cache_lock:
                    if key in images_bgr:
                        continue
                with pending_lock:
                    if key in pending:
                        continue
                    pending[key] = executor.submit(_get_image, pp)

        def _wait_prefetch(batch_paths: List[Path]) -> None:
            if executor is None:
                return
            for pp in batch_paths:
                key = pp.name
                with pending_lock:
                    fut = pending.pop(key, None)
                if fut is not None:
                    fut.result()

        if EAGER_PREFETCH_ALL:
            _submit_prefetch(images)

        for roi_idx, roi in enumerate(rois):
            roi_tiles = 0
            roi_prep_s = 0.0
            roi_ocr_s = 0.0
            roi_crop_s = 0.0
            batch_idx = 0

            for start in range(0, len(images), STITCH_BATCH_SIZE):
                batch = images[start : start + STITCH_BATCH_SIZE]

                # Wait any prefetched reads for current batch.
                _wait_prefetch(batch)

                tiles: List[np.ndarray] = []
                names: List[str] = []

                prep_t0 = time.perf_counter()
                for p in batch:
                    img = _get_image(p)
                    c0 = time.perf_counter()
                    tile = crop_from_image(img, roi, ROI_PADDING)
                    c1 = time.perf_counter()
                    roi_crop_s += c1 - c0
                    tiles.append(tile)
                    names.append(p.name)
                prep_t1 = time.perf_counter()

                # Start prefetch for next batch while OCR runs on current batch.
                if not EAGER_PREFETCH_ALL:
                    next_batch = images[start + STITCH_BATCH_SIZE : start + 2 * STITCH_BATCH_SIZE]
                    _submit_prefetch(next_batch)

                batch_label = f"{frames_dir} roi#{roi_idx} [{start}:{start+len(batch)}]"
                ocr_t0 = time.perf_counter()
                timing_detail: Dict[str, float] | None = {} if (TIMING_ENABLED and TIMING_DETAILED) else None
                temps_list, raw_list = _ocr_mosaic_tiles(tiles, names, ocr_engine, backend, batch_label, timing_detail)
                ocr_t1 = time.perf_counter()

                roi_tiles += len(batch)
                roi_prep_s += prep_t1 - prep_t0
                roi_ocr_s += ocr_t1 - ocr_t0

                if TIMING_ENABLED and TIMING_PRINT_EVERY_BATCH > 0 and (batch_idx % TIMING_PRINT_EVERY_BATCH == 0):
                    total_s = (prep_t1 - prep_t0) + (ocr_t1 - ocr_t0)
                    per_tile = total_s / max(len(batch), 1)
                    if timing_detail is not None:
                        mosaic_s = timing_detail.get("mosaic_s", 0.0)
                        ocr_items_s = timing_detail.get("ocr_items_s", 0.0)
                        fallback_s = timing_detail.get("fallback_s", 0.0)
                        print(
                            f"[timing] {batch_label} prep={prep_t1 - prep_t0:.3f}s "
                            f"ocr={ocr_t1 - ocr_t0:.3f}s total={total_s:.3f}s "
                            f"({len(batch)} tiles, {per_tile:.4f}s/tile) "
                            f"| mosaic={mosaic_s:.3f}s ocr_items={ocr_items_s:.3f}s fallback={fallback_s:.3f}s"
                        )
                    else:
                        print(
                            f"[timing] {batch_label} prep={prep_t1 - prep_t0:.3f}s "
                            f"ocr={ocr_t1 - ocr_t0:.3f}s total={total_s:.3f}s "
                            f"({len(batch)} tiles, {per_tile:.4f}s/tile)"
                        )
                batch_idx += 1

                for name, temp, raw in zip(names, temps_list, raw_list, strict=True):
                    temps_map[name][roi_idx] = temp
                    raw_map[name][roi_idx] = raw

            if TIMING_ENABLED:
                total = roi_prep_s + roi_ocr_s
                tps = roi_tiles / total if total > 0 else 0.0
                if TIMING_DETAILED:
                    print(
                        f"[timing] {frames_dir} roi#{roi_idx} summary: tiles={roi_tiles} "
                        f"prep={roi_prep_s:.3f}s (crop={roi_crop_s:.3f}s) ocr={roi_ocr_s:.3f}s "
                        f"total={total:.3f}s ({tps:.1f} tiles/s)"
                    )
                else:
                    print(
                        f"[timing] {frames_dir} roi#{roi_idx} summary: tiles={roi_tiles} "
                        f"prep={roi_prep_s:.3f}s ocr={roi_ocr_s:.3f}s total={total:.3f}s ({tps:.1f} tiles/s)"
                    )

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    for p in images:
        results.append((p.name, temps_map[p.name], raw_map[p.name]))

    write_csv(results, output_csv, roi_count=roi_count)

    if TIMING_ENABLED:
        folder_t1 = time.perf_counter()
        if TIMING_DETAILED:
            total_s = folder_t1 - folder_t0
            print(
                f"[timing] {frames_dir} total elapsed: {total_s:.3f}s "
                f"| imread={imread_s_total:.3f}s (count={imread_count}, cache_hits={cache_hits})"
            )
        else:
            print(f"[timing] {frames_dir} total elapsed: {folder_t1 - folder_t0:.3f}s")


def main() -> None:
    rois = load_rois(ROI_CONFIG)
    backend = BACKEND.lower().strip()
    ocr_engine = _init_ocr_engine(backend)

    experiment_dir = EXPERIMENT_DIR
    if not experiment_dir.exists():
        print(f"Skip missing experiment dir: {experiment_dir}")
        return

    folders = _find_frames_folders(experiment_dir)
    if not folders:
        print(f"No subfolders with '{FRAMES_SUBDIR}' found under: {experiment_dir}")
        return

    for folder in folders:
        frames_dir = folder / FRAMES_SUBDIR
        out_csv = folder / OUTPUT_CSV_NAME
        print(f"Processing {experiment_dir.name} {folder.name} -> {out_csv}")
        _process_frames_folder(frames_dir, out_csv, rois, ocr_engine, backend)

    print("Done.")


if __name__ == "__main__":
    main()
