"""
Build a small dev copy of your dataset, optionally binned into class folders.

Reads {train,val,test}_annotations_cropped.csv from --src, samples N images per class
(or takes every row with --no-sample), copies them under --dst using the
chosen --layout, and writes new CSVs in --dst whose path column matches the
new locations. Rows whose image file can't be found are dropped (with a warning).

Layouts (what the copied files look like under --dst):
  preserve     same relative paths as the source CSV   (default)
  class        <dst>/<class_name>/<file>
  split-class  <dst>/<split>/<class_name>/<file>      e.g. train/zebra_plains/x.jpg
  flat         <dst>/imgs/<file>                        (single flat folder)

If you already generated subset CSVs whose paths still point at the original
images, use them as --src with --no-sample and give --img-root as the folder
that holds the original images.

Example:
    python make_dev_subset_v1.py --src ./data/gzgc_dataset/dev_crops/cropped_resized --img-root ./data/gzgc_dataset/dev_crops/cropped_resized \
    --dst ./data/gzgc_dataset/dev_crops_subset/cropped_resized --class-col classification --label-col label --layout flat \
    --train-per-class 20 --val-per-class 8 --test-per-class 8
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


def safe_name(s):
    return re.sub(r"[^\w\-.]+", "_", str(s)).strip("_")


def subset_split(df, class_col, n_per_class, seed):
    parts = [g.sample(n=min(n_per_class, len(g)), random_state=seed)
             for _, g in df.groupby(class_col)]
    return pd.concat(parts).sort_index()


def target_rel(layout, split, label, old_rel):
    name = Path(old_rel).name
    if layout == "preserve":
        return Path(old_rel)
    if layout == "class":
        return Path(safe_name(label)) / name
    if layout == "split-class":
        return Path(split) / safe_name(label) / name
    if layout == "flat":
        return Path(name)
    raise ValueError(layout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="folder with the *_annotations_cropped.csv files")
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--img-root", type=Path, default=None,
                    help="folder CSV image paths are relative to (default: --src)")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--class-col", default="classification")
    ap.add_argument("--label-col", default="label",
                    help="column used for class folder names (falls back to --class-col)")
    ap.add_argument("--layout", choices=["preserve", "class", "split-class", "flat"],
                    default="preserve")
    ap.add_argument("--no-sample", action="store_true", help="use every row instead of sampling")
    ap.add_argument("--train-per-class", type=int, default=20)
    ap.add_argument("--val-per-class", type=int, default=8)
    ap.add_argument("--test-per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img_root = args.img_root or args.src
    dst = args.dst.resolve()
    if dst in (args.src.resolve(), img_root.resolve()):
        sys.exit("--dst must be a different folder from --src and --img-root.")
    dst.mkdir(parents=True, exist_ok=True)

    sizes = {"train": args.train_per_class, "val": args.val_per_class,
             "test": args.test_per_class}
    all_classes = None

    for split, n in sizes.items():
        csv_in = args.src / f"{split}_annotations_cropped.csv"
        print(f"{csv_in.name}: {n}")
        if not csv_in.exists():
            print(f"[skip] {csv_in} not found")
            continue

        df = pd.read_csv(csv_in)
        for col in (args.path_col, args.class_col):
            if col not in df.columns:
                sys.exit(f"{csv_in.name}: no column '{col}'. Columns found: {list(df.columns)}")
        label_col = args.label_col if args.label_col in df.columns else args.class_col

        if all_classes is None:
            all_classes = set(df[args.class_col].dropna().unique().tolist())
        sub = df if args.no_sample else subset_split(df, args.class_col, n, args.seed)

        keep, new_paths, used = [], [], set()
        printed_missing = False
        for idx, row in sub.iterrows():
            old_rel = str(row[args.path_col]).replace("\\", "/")
            src_file = img_root / old_rel
            if not src_file.exists():
                if not printed_missing:
                    print(f"  first missing file looked for at: {src_file}")
                    printed_missing = True
                continue
            rel = target_rel(args.layout, split, row[label_col], old_rel)
            k = 1
            while rel in used:  # same filename from different folders
                rel = rel.with_name(f"{Path(old_rel).stem}_{k}{Path(old_rel).suffix}")
                k += 1
            used.add(rel)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst / rel)
            keep.append(idx)
            new_paths.append(rel.as_posix())

        out = sub.loc[keep].copy()
        out[args.path_col] = new_paths
        out.to_csv(dst / f"{split}_annotations_cropped.csv", index=False)

        dropped = len(sub) - len(out)
        counts = out[args.class_col].value_counts().sort_index().to_dict()
        print(f"[{split}] {len(out)} images copied  per-class: {counts}")
        if dropped:
            print(f"  WARNING: {dropped} rows dropped (image not found under {img_root})")
        absent = all_classes - set(out[args.class_col].unique())
        if absent:
            print(f"  WARNING: classes missing from this split: {sorted(absent)}")

    print(f"\nDev subset written to {dst}")


if __name__ == "__main__":
    main()
