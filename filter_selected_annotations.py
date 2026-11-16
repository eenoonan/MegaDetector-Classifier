#!/usr/bin/env python3
# filter_selected_annotations.py
"""
For each set of folders
    baseline -> bl_selected
    sil      -> sil_selected
    swatch   -> swatch_selected
read test_annotations_cropped.csv from the full folder and write a filtered copy
to the *_selected folder, keeping only rows whose image is actually in that
selected folder. Header, columns and delimiter (tab or comma) are preserved.

Usage:
    python filter_selected_annotations.py --root path/to/cropped_resized

Folder layout expected under --root:
    baseline/       test_annotations_cropped.csv
    bl_selected/   (images)             -> gets test_annotations_cropped.csv
    sil/            test_annotations_cropped.csv
    sil_selected/   (images)             -> gets test_annotations_cropped.csv
    swatch/         test_annotations_cropped.csv
    swatch_selected/ (images)            -> gets test_annotations_cropped.csv

Optional:
    --csv-name NAME        annotations file name (default test_annotations_cropped.csv)
    --filename-col NAME    column holding image file names (auto-detected if omitted)
    --update-path          also rewrite the 'path' column to point at the *_selected folder
"""
import argparse
import csv
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
PAIRS = [("baseline", "bl_selected"),("sil", "sil_selected"), ("swatch", "swatch_selected")]


def _stem(value: str) -> str:
    """Lower-case file stem; treats both / and \\ as folder separators."""
    return Path(value.strip().replace("\\", "/")).stem.lower()


def filter_csv(src_csv: Path, selected_dir: Path, forced_col=None,
               update_path=False, path_col="path"):
    images = {p.stem.lower(): p.name for p in selected_dir.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    if not images:
        print(f"[warn] no images found in {selected_dir}")

    text = src_csv.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    delim = max(",\t;", key=lines[0].count)
    table = list(csv.reader(lines, delimiter=delim))
    header, rows = table[0], table[1:]

    # Which column holds the image file names?
    if forced_col:
        if forced_col not in header:
            raise SystemExit(f"Column '{forced_col}' not in header: {header}")
        col = header.index(forced_col)
    else:
        hits = [sum(1 for r in rows if i < len(r) and _stem(r[i]) in images)
                for i in range(len(header))]
        if max(hits, default=0) == 0:
            raise SystemExit(f"No column in {src_csv} matches the images in "
                             f"{selected_dir}; try --filename-col.")
        col = hits.index(max(hits))  # first best column wins ties

    pcol = header.index(path_col) if update_path and path_col in header else None

    kept, seen = [], set()
    for r in rows:
        if col < len(r) and _stem(r[col]) in images:
            r = list(r)
            seen.add(_stem(r[col]))
            if pcol is not None and pcol < len(r) and r[pcol].strip():
                old = r[pcol].strip()
                sep = "\\" if "\\" in old else "/"
                parent = old.replace("\\", "/").rsplit("/", 1)[0].split("/")
                parent[-1] = selected_dir.name  # swap sil -> sil_selected etc.
                r[pcol] = sep.join(parent + [images[_stem(r[col])]])
            kept.append(r)

    out = selected_dir / src_csv.name
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delim, lineterminator="\n")
        w.writerow(header)
        w.writerows(kept)

    missing = sorted(set(images) - seen)
    print(f"[ok]   {out}: {len(kept)} rows kept of {len(rows)} "
          f"({len(images)} images in folder, column '{header[col]}')")
    if missing:
        print(f"[warn] {len(missing)} image(s) had no row in {src_csv.name}: "
              f"{', '.join(images[m] for m in missing[:5])}"
              f"{' ...' if len(missing) > 5 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="folder containing sil, sil_selected, swatch, swatch_selected")
    ap.add_argument("--csv-name", default="test_annotations_cropped.csv")
    ap.add_argument("--filename-col", default=None)
    ap.add_argument("--update-path", action="store_true",
                    help="rewrite the 'path' column to point at the *_selected folder")
    args = ap.parse_args()

    for full, selected in PAIRS:
        src = args.root / full / args.csv_name
        sel_dir = args.root / selected
        if not src.is_file():
            print(f"[skip] {src} not found")
            continue
        if not sel_dir.is_dir():
            print(f"[skip] {sel_dir} not found")
            continue
        filter_csv(src, sel_dir, args.filename_col, args.update_path)


if __name__ == "__main__":
    main()
