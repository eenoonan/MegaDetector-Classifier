#!/usr/bin/env python3
"""
Batch-create a silhouette and a fur swatch for each animal JPG.

Install:
    pip install rembg onnxruntime opencv-python pillow numpy

Usage:
    python animal_batch.py --input ./path/to/cropped_images
    python animal_batch.py --input ./path/to/cropped_images --swatch-size 512 --erode 15

    python animal_batch.py --input ./data/gzgc_dataset/eval_crops/cropped_resized \
    --annot ./data/gzgc_dataset/eval_crops/annot_eval_crop.csv --swatch-size 100 --erode 15

Outputs go under --output (default: ./data/gzgc_dataset/eval_crops/cropped_resized), in three folders.
For an input "lion.jpg":
    baseline/lion.jpg   copy of the original cropped image
    sil/lion.png        black animal on white background
    swatch/lion.jpg     square crop of fur, resized to --swatch-size
Every version keeps the original file name (silhouettes use .png so the pure
black/white values aren't blurred by JPEG compression).
"""
import csv
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove


def get_mask(img: Image.Image, session) -> np.ndarray | None:
    """Return a clean binary mask (255 = animal) using background removal."""
    rgba = remove(img, session=session)  # PIL RGBA, background transparent
    alpha = np.array(rgba)[:, :, 3]
    mask = (alpha > 128).astype(np.uint8) * 255

    # Keep only the largest connected blob (drops stray specks)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = ((labels == largest) * 255).astype(np.uint8)

    # Smooth ragged edges and close tiny gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def make_silhouette(mask: np.ndarray) -> Image.Image:
    """Black animal on white background. Fills interior holes (e.g. eyes)."""
    filled = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    out = np.where(filled > 0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def make_swatch(img_rgb: np.ndarray, mask: np.ndarray, size: int, erode_px: int):
    """
    Find the largest axis-aligned square that fits entirely inside the
    (eroded) animal mask, crop it from the photo, and resize to size x size.
    Returns (PIL image, side_in_pixels) or (None, 0).
    """
    # Erode so the square stays away from the animal's outline / background bleed
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
        inner = cv2.erode(mask, k)
        if inner.max() == 0:  # animal too small for that erosion
            inner = mask
    else:
        inner = mask

    # Chebyshev distance transform: value at a pixel = half-side of the largest
    # square centred there that stays inside the mask. Pad so the image border
    # counts as "outside".
    padded = cv2.copyMakeBorder(inner, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_C, 3)
    _, max_d, _, (cx, cy) = cv2.minMaxLoc(dist)
    cx, cy = cx - 1, cy - 1  # undo padding offset
    half = int(max_d) - 1
    if half < 8:
        return None, 0

    crop = img_rgb[cy - half : cy + half + 1, cx - half : cx + half + 1]
    swatch = Image.fromarray(crop).resize((size, size), Image.LANCZOS)
    return swatch, 2 * half + 1

def _stem(value):
    """Lower-case file stem, treating both / and \\ as folder separators."""
    return Path(value.strip().replace("\\", "/")).stem.lower()


def find_filename_col(header, rows, stems, forced=None):
    """Index of the column holding image file names (auto-detected unless forced)."""
    if forced is not None:
        if forced not in header:
            raise SystemExit(f"--filename-col '{forced}' not in CSV header: {header}")
        return header.index(forced)
    best, best_hits = None, 0
    for i in range(len(header)):
        hits = sum(1 for r in rows if i < len(r) and _stem(r[i]) in stems)
        if hits > best_hits:  # strict '>' so the first matching column wins ties
            best, best_hits = i, hits
    if best is None:
        raise SystemExit("Could not find a column matching the image file names; "
                         "use --filename-col to name it.")
    return best


def write_annotations(annot_path, folder, produced, forced_col=None, path_col="path"):
    """
    Write a copy of the annotations file into `folder`, keeping the header, the
    delimiter (tab or comma) and every column. Rows are kept only for images
    that exist in that folder. Two columns are updated per row:
      - the file-name column (default: auto-detected, e.g. 'crop'): new file name
        (e.g. .jpg -> .png for silhouettes)
      - `path_col` (if present): the original folder path with the sub-folder
        (baseline / sil / swatch) and new file name appended, same slash style.
    `produced` maps original stem (lowercase) -> new file name in `folder`.
    """
    with open(annot_path, newline="", encoding="utf-8-sig") as f:
        text = f.read()
    first_line = text.splitlines()[0]
    delim = max(",\t;", key=first_line.count)
    table = list(csv.reader(text.splitlines(), delimiter=delim))
    header, rows = table[0], table[1:]
    col = find_filename_col(header, rows, set(produced), forced_col)
    pcol = header.index(path_col) if path_col in header else None

    out_rows = []
    for r in rows:
        if col >= len(r) or _stem(r[col]) not in produced:
            continue
        new_name = produced[_stem(r[col])]
        r = list(r)
        old_value = r[col].strip()
        r[col] = new_name if Path(old_value.replace("\\", "/")).suffix \
            else Path(new_name).stem
        if pcol is not None and pcol < len(r) and r[pcol].strip():
            old_path = r[pcol].strip()
            sep = "\\" if "\\" in old_path else "/"
            parent = old_path.replace("\\", "/").rsplit("/", 1)[0]
            r[pcol] = sep.join(parent.split("/") + [folder.name, new_name])
        out_rows.append(r)

    with open(folder / Path(annot_path).name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delim, lineterminator="\n")
        w.writerow(header)
        w.writerows(out_rows)
    return len(out_rows), len(rows), header[col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("./data/gzgc_dataset/eval_crops/cropped_resized"),
                    help="parent folder that will hold baseline/, sil/ and swatch/")
    ap.add_argument("--annot", type=Path, default=None,
                    help="path to annot_eval_crops.csv; a matching CSV is written to each folder")
    ap.add_argument("--filename-col", default=None,
                    help="CSV column holding image file names (auto-detected if omitted)")
    ap.add_argument("--path-col", default="path",
                    help="CSV column holding the image path (default: 'path')")
    ap.add_argument("--swatch-size", type=int, default=256)
    ap.add_argument("--erode", type=int, default=10,
                    help="pixels to pull the swatch away from the animal's edge")
    ap.add_argument("--model", default="isnet-general-use",
                    help="rembg model (try u2net, isnet-general-use, birefnet-general)")
    args = ap.parse_args()

    if args.annot is not None:
        if args.annot.is_dir():
            found = sorted(p for p in args.annot.iterdir()
                           if p.is_file() and "annot" in p.name.lower()
                           and p.suffix.lower() in {".csv", ".tsv", ".txt"})
            if len(found) != 1:
                raise SystemExit(
                    f"--annot points to a folder ({args.annot}). Give the path to the "
                    f"annotations file itself, e.g. {args.annot / 'annot_eval_crops.csv'}")
            args.annot = found[0]
            print(f"Using annotations file: {args.annot}")
        elif not args.annot.is_file():
            raise SystemExit(f"--annot file not found: {args.annot}")

    baseline_dir = args.output / "baseline"
    sil_dir = args.output / "sil"
    swatch_dir = args.output / "swatch"
    for d in (baseline_dir, sil_dir, swatch_dir):
        d.mkdir(parents=True, exist_ok=True)
    session = new_session(args.model)  # load once, reuse for every image

    files = sorted(p for p in args.input.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg"})
    print(f"Found {len(files)} images")

    # original stem (lowercase) -> file name written in each folder
    baseline_ok, sil_ok, swatch_ok = {}, {}, {}

    for path in files:
        key = path.stem.lower()
        try:
            # Baseline: untouched copy of the original cropped image
            shutil.copy2(path, baseline_dir / path.name)
            baseline_ok[key] = path.name

            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            mask = get_mask(img, session)
            if mask is None:
                print(f"[skip] {path.name}: no animal detected (baseline copied only)")
                continue

            make_silhouette(mask).save(sil_dir / f"{path.stem}.png")
            sil_ok[key] = f"{path.stem}.png"

            swatch, side = make_swatch(np.array(img), mask, args.swatch_size, args.erode)
            if swatch is None:
                print(f"[warn] {path.name}: silhouette saved, but no usable swatch area")
                continue
            swatch.save(swatch_dir / f"{path.stem}.jpg", quality=95)
            swatch_ok[key] = f"{path.stem}.jpg"

            note = "  (small source patch, swatch is upscaled)" if side < args.swatch_size else ""
            print(f"[ok]   {path.name}: swatch from {side}px square{note}")
        except Exception as e:
            print(f"[fail] {path.name}: {e}")

    # Annotation CSVs: same columns/format as the original, one per folder
    if args.annot:
        for folder, produced in ((baseline_dir, baseline_ok),
                                 (sil_dir, sil_ok),
                                 (swatch_dir, swatch_ok)):
            kept, total, col = write_annotations(args.annot, folder, produced,
                                                 args.filename_col, args.path_col)
            print(f"[csv]  {folder.name}/{args.annot.name}: {kept}/{total} rows "
                  f"(file-name column: '{col}')")

if __name__ == "__main__":
    main()
