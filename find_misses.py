# find_misses.py
"""
Per-image predictions for one dataset split, plus a folder of the misclassified
images. Put this file next to main.py and run it from the same folder.

It builds the same dataset/learner as main.py (so image transforms and class
mapping match training), loads the checkpoint, and runs the images through
net.feature -> net.classifier on CPU.

Outputs in --out-dir:
  predictions_<name>.csv   every image: true/pred label, confidence, correct
  misses_<name>.csv        only the wrong ones, most confident mistakes first
  misses_<name>/<true>_predicted_as_<pred>/<conf>_<file>   copies for viewing

Examples:
  # dev test split (the config you trained with)
  python find_misses.py --config configs/config.yaml --ckpt <ckpt> --split test --name dev_test
  # validation split (the ones that came back 99.x%)
  python find_misses.py --config configs/config.yaml --ckpt <ckpt> --split val --name dev_val
  # an eval site, using the config run_predict.py generated
  python find_misses.py --config eval_results/config_siteB.yaml --ckpt <ckpt> --name siteB
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from munch import Munch
from PIL import Image

from src import algorithms, datasets


class ImageList(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--name", default="run")
    ap.add_argument("--out-dir", default="miss_results", type=Path)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-copy", action="store_true", help="skip copying missed images")
    ap.add_argument("--max-copy", type=int, default=300)
    args = ap.parse_args()

    with open(args.config) as f:
        conf = Munch(yaml.load(f, Loader=yaml.FullLoader))
    conf.evaluate, conf.val, conf.test = args.ckpt, False, True
    conf.predict, conf.predict_root = False, ""

    dataset = datasets.__dict__[conf.dataset_name](conf=conf)
    counts = getattr(dataset, "train_class_counts", None)
    if counts is None:  # only used for loss weighting, irrelevant at inference
        counts = [1] * conf.num_classes
    learner = algorithms.__dict__[conf.algorithm](
        conf=conf, train_class_counts=counts, id_to_labels=dataset.id_to_labels)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    learner.load_state_dict(ck["state_dict"])
    net = learner.net
    if not (hasattr(net, "feature") and hasattr(net, "classifier")):
        sys.exit(f"learner.net has no feature/classifier parts; it has: {list(dict(net.named_children()))}")
    net.eval()

    # same transform the dataset uses for evaluation
    tf = getattr(getattr(dataset, "dset_te", None), "transform", None)
    if tf is None:
        from torchvision import transforms as T
        print("WARNING: could not read the dataset's transform; using a generic "
              "224px ImageNet transform. Compare the accuracy printed below with "
              "the one from main.py --test to make sure they agree.")
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    root = Path(conf.dataset_root) / "cropped_resized"
    csv = root / f"{args.split}_annotations_cropped.csv"
    ann = pd.read_csv(csv)
    names = ann["path"].astype(str).str.replace("\\", "/", regex=False)
    paths = [root / n for n in names]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} images listed in {csv} not found, e.g. {missing[:3]}")

    loader = torch.utils.data.DataLoader(ImageList(paths, tf), batch_size=args.batch_size, shuffle=False)
    outs = []
    with torch.no_grad():
        for x in loader:
            feats = net.feature(x)
            if feats.ndim > 2:
                feats = torch.flatten(feats, 1)
            outs.append(torch.softmax(net.classifier(feats), dim=1))
            print(f"\r{sum(len(o) for o in outs)}/{len(paths)} images", end="", flush=True)
    print()
    probs = torch.cat(outs).numpy()

    id_to_label = {int(k): str(v) for k, v in dataset.id_to_labels.items()}
    true = ann["classification"].astype(int).to_numpy()
    pred = probs.argmax(1)
    res = pd.DataFrame({
        "path": names,
        "true_id": true, "true_label": [id_to_label.get(t, str(t)) for t in true],
        "pred_id": pred, "pred_label": [id_to_label.get(p, str(p)) for p in pred],
        "confidence": probs.max(1),
        "true_class_prob": probs[np.arange(len(true)), true],
    })
    for k in range(probs.shape[1]):  # full probability for every class (needed for ROC / PR)
        res[f"prob_{id_to_label.get(k, k)}"] = probs[:, k]
    res["correct"] = res["true_id"] == res["pred_id"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_dir / f"predictions_{args.name}.csv", index=False)
    miss = res[~res["correct"]].sort_values("confidence", ascending=False)
    miss.to_csv(args.out_dir / f"misses_{args.name}.csv", index=False)

    per_class = res.groupby("true_label")["correct"].mean()
    print(f"\n{args.name} [{args.split}]  images: {len(res)}  missed: {len(miss)}")
    print(f"accuracy: {res['correct'].mean():.4f}   macro (mean per-class) accuracy: {per_class.mean():.4f}")
    print("per-class accuracy:", per_class.round(4).to_dict())
    print("confusion (rows = true, cols = predicted):")
    print(pd.crosstab(res["true_label"], res["pred_label"]))

    if len(miss) and not args.no_copy:
        base = args.out_dir / f"misses_{args.name}"
        for (_, r), p in zip(miss.head(args.max_copy).iterrows(), [root / n for n in miss["path"].head(args.max_copy)]):
            d = base / f"{r.true_label}_predicted_as_{r.pred_label}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d / f"{r.confidence:.3f}_{Path(r.path).name}")
        print(f"\nCopied {min(len(miss), args.max_copy)} missed images to {base}")
    print(f"CSVs written to {args.out_dir}")


if __name__ == "__main__":
    main()

