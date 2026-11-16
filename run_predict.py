# run_predict.py
"""
Evaluate one fine-tuned checkpoint on several held-out datasets by calling the
(patched) main.py once per dataset in --test mode on CPU.

Each dataset folder must contain cropped_resized/test_annotations_cropped.csv
(path, classification, label) plus the images in cropped_resized/, which is
the layout the repo's Custom_Crop dataset reads.

custom.py requires:   <dataset_root>/cropped_resized/test_annotations_cropped.csv
                      <dataset_root>/cropped_resized/<images>
This script builds that layout for you. For each NAME=IMAGE_FOLDER it:
  1. reads the eval annotation CSV (default: <IMAGE_FOLDER>/../../atest_annotations_cropped.csv,
     override with --ann),
  2. keeps the rows whose image file exists in IMAGE_FOLDER,
  3. hard-links (or copies) those images into
        <out-dir>/staging/NAME/cropped_resized/
     and writes test_annotations_cropped.csv there (path = bare filename),
  4. points a copy of the base config at <out-dir>/staging/NAME and runs main.py.
If IMAGE_FOLDER already contains cropped_resized/test_annotations_cropped.csv,
it is used as-is and nothing is staged.


Example:
    python run_predict.py \
        --base-config configs/config.yaml \
        --ckpt weights/<log_dir>/Plain/<your>.ckpt \
        --datasets siteA=/data/evalA siteB=/data/evalB siteC=/data/evalC \
        --out-dir eval_results

    python run_predict.py \
        --base-config ./configs/config.yaml \
        --ckpt ./weights/Crop/Plain/Crop_Res18_plain_071824-0-epoch=01-valid_mac_acc=100.00-v1.ckpt \
        --datasets siteA=./data/gzgc_dataset/eval_crops/cropped_resized/baseline \
        siteB=./data/gzgc_dataset/eval_crops/cropped_resized/sil siteC=./data/gzgc_dataset/eval_crops/cropped_resized/swatch \
        --out-dir eval_results
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def stage_dataset(name, img_dir, ann_csv, out_dir):
    df = pd.read_csv(ann_csv)
    for col in ("classification", "label"):
        if col not in df.columns:
            sys.exit(f"{ann_csv}: missing column '{col}'. Columns: {list(df.columns)}")
    name_col = "crop" if "crop" in df.columns else "path"
    csv_names = df[name_col].astype(str).str.replace("\\", "/", regex=False).map(lambda s: Path(s).name)

    # index every image under img_dir (subfolders included)
    files = [p.relative_to(img_dir) for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
    by_name, by_stem = {}, {}
    for rel in files:
        by_name.setdefault(rel.name, rel)
        by_stem.setdefault(rel.stem, rel)
    stems = sorted(by_stem)

    def find(fn):
        p = Path(fn)
        if p.name in by_name:
            return by_name[p.name], "exact name"
        if p.stem in by_stem:
            return by_stem[p.stem], "name ignoring extension"
        cands = [t for t in stems if t.startswith(p.stem)]
        if len(cands) == 1:
            return by_stem[cands[0]], "unique filename prefix"
        return None, None

    matches = [find(fn) for fn in csv_names]
    rels = [m[0] for m in matches]
    modes = pd.Series([m[1] for m in matches]).value_counts()
    ok = [r is not None for r in rels]
    if not any(ok):
        print(f"[{name}] no match. {len(files)} images found under {img_dir}")
        print("  CSV names, e.g.:   ", list(csv_names[:3]))
        print("  Folder files, e.g.:", [str(f) for f in files[:3]])
        sys.exit(1)

    sub = df[ok].copy()
    sub["path"] = [r.as_posix() for r in rels if r is not None]
    print(f"[{name}] {len(sub)} of {len(df)} annotated images found in {img_dir}")
    print("  match types:", modes.to_dict())
    print("  e.g.", list(zip(csv_names[ok][:2], sub["path"][:2])))

    stage_root = out_dir / "staging" / name
    crops = stage_root / "cropped_resized"
    crops.mkdir(parents=True, exist_ok=True)
    for f in sub["path"]:
        dst = crops / f
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(img_dir / f, dst)  # no extra disk space
        except OSError:
            shutil.copy2(img_dir / f, dst)
    sub.to_csv(crops / "test_annotations_cropped.csv", index=False)
    return stage_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--datasets", required=True, nargs="+", help="name=/path/to/image/folder")
    ap.add_argument("--ann", type=Path, default=None,
                    help="eval annotation CSV (default: <folder>/../../test_annotations_cropped.csv)")
    ap.add_argument("--out-dir", default=Path("eval_results"), type=Path)
    ap.add_argument("--main", default="main.py")
    ap.add_argument("--gpus", default="cpu")
    ap.add_argument("--np-threads", default="4")
    args = ap.parse_args()

    if not args.ckpt.exists():
        sys.exit(f"Checkpoint not found: {args.ckpt.resolve()}\n"
                 "Use a path relative to this folder (e.g. ./weights/...), not one "
                 "starting with a bare '/'.")
    args.ckpt = args.ckpt.resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(args.base_config.read_text())
    summary = {}

    for spec in args.datasets:
        name, path = spec.split("=", 1)
        path = Path(path)
        print(f"Annot Path: {path}")
        if (path / "cropped_resized" / "test_annotations_cropped.csv").exists():
            root = path  # already in the layout custom.py expects
        else:
            ann = args.ann or path.parent.parent / "annot_eval_crops.csv"
            if not ann.exists():
                sys.exit(f"{ann} not found (use --ann to point at the eval annotation CSV)")
            root = stage_dataset(name, path, ann, args.out_dir)

        conf = dict(base)
        conf["dataset_root"] = str(root)
        conf["annotation_dir"] = str(root)
        conf["split_data"] = False
        cfg_path = args.out_dir / f"config_{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(conf))

        out_json = args.out_dir / f"{name}_metrics.json"
        cmd = [sys.executable, args.main, "--config", str(cfg_path),
               "--evaluate", str(args.ckpt), "--test",
               "--gpus", args.gpus, "--np-threads", args.np_threads,
               "--out-json", str(out_json)]
        print(f"\n=== {name} ===\n{' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        summary[name] = json.loads(out_json.read_text())

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    for name, res in summary.items():
        print(name, res[0] if res else res)


if __name__ == "__main__":
    main()
