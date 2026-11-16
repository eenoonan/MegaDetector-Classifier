# plot_metrics.py
"""
Confusion matrix, ROC and precision-recall plots from predictions_<name>.csv
files written by find_misses.py (they hold the true label, predicted label and
a prob_<class> column for every class).

For each file it writes, into --out-dir:
  <name>_confusion.png        counts and row-normalised side by side
  <name>_roc.png              ROC curve(s) with AUC
  <name>_pr.png               precision-recall curve(s) with average precision
  <name>_report.csv           per-class precision / recall / F1 / support
And when several files are given:
  comparison_roc.png, comparison_pr.png   all sites overlaid (two-class only)
  comparison_summary.csv                  one row per site

Two classes: curves are for --positive (default: the class with the higher id).
More than two: one-vs-rest curve per class.

Example:
    python plot_metrics.py miss_results/predictions_siteA.csv \
        miss_results/predictions_siteB.csv miss_results/predictions_siteC.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)


def load(path):
    df = pd.read_csv(path)
    # prob_of_true_class (written by older find_misses.py) is not a class
    classes = [c[5:] for c in df.columns
               if c.startswith("prob_") and c != "prob_of_true_class"]
    if not classes:
        raise SystemExit(f"{path}: no prob_<class> columns. Re-run the updated find_misses.py.")
    return df, classes


def positive_class(df, classes, requested):
    if requested:
        if requested not in classes:
            raise SystemExit(f"--positive {requested!r} not in {classes}")
        return requested
    ids = df.drop_duplicates("true_label").set_index("true_label")["true_id"].to_dict()
    ids.update(df.drop_duplicates("pred_label").set_index("pred_label")["pred_id"].to_dict())
    return max(classes, key=lambda c: ids.get(c, -1))


def plot_confusion(df, classes, out, title):
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=classes)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, fmt, sub in [(axes[0], cm, "d", "Counts"), (axes[1], norm, ".1%", "Row-normalised")]:
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max() if data.max() else 1)
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(sub)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v > data.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def curves(df, classes, pos):
    """yield (label, y_true_binary, score) for the positive class or one-vs-rest."""
    targets = [pos] if len(classes) == 2 else classes
    for c in targets:
        y = (df["true_label"] == c).astype(int).to_numpy()
        yield c, y, df[f"prob_{c}"].to_numpy()


def plot_roc_pr(df, classes, pos, out_roc, out_pr, title):
    fr, ar = plt.subplots(figsize=(5.2, 5))
    fp, ap = plt.subplots(figsize=(5.2, 5))
    stats = {}
    for c, y, sc in curves(df, classes, pos):
        if y.min() == y.max():
            print(f"  [{title}] skipping '{c}': only one class present in the true labels")
            continue
        fpr, tpr, _ = roc_curve(y, sc)
        prec, rec, _ = precision_recall_curve(y, sc)
        auc, ap_ = roc_auc_score(y, sc), average_precision_score(y, sc)
        stats[c] = (auc, ap_)
        ar.plot(fpr, tpr, label=f"{c} (AUC {auc:.4f})")
        ap.plot(rec, prec, label=f"{c} (AP {ap_:.4f})")
    ar.plot([0, 1], [0, 1], "k--", lw=0.8)
    ar.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC: {title}")
    ap.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall: {title}", ylim=(0, 1.02))
    for a, f, o in [(ar, fr, out_roc), (ap, fp, out_pr)]:
        a.legend(loc="lower right" if a is ar else "lower left")
        f.tight_layout()
        f.savefig(o, dpi=150)
        plt.close(f)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+", type=Path)
    ap.add_argument("--out-dir", default=Path("plots"), type=Path)
    ap.add_argument("--positive", default=None, help="positive class for two-class ROC/PR")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, overlay = [], {}
    for f in args.predictions:
        name = f.stem.replace("predictions_", "")
        df, classes = load(f)
        pos = positive_class(df, classes, args.positive)

        plot_confusion(df, classes, args.out_dir / f"{name}_confusion.png", name)
        stats = plot_roc_pr(df, classes, pos, args.out_dir / f"{name}_roc.png",
                            args.out_dir / f"{name}_pr.png", name)
        rep = classification_report(df["true_label"], df["pred_label"], labels=classes,
                                    output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(args.out_dir / f"{name}_report.csv")

        per_class_recall = [rep[c]["recall"] for c in classes]
        rows.append({
            "dataset": name, "n": len(df),
            "accuracy": (df["true_label"] == df["pred_label"]).mean(),
            "macro_accuracy": float(np.mean(per_class_recall)),
            "macro_f1": f1_score(df["true_label"], df["pred_label"], labels=classes, average="macro", zero_division=0),
            "roc_auc": np.mean([v[0] for v in stats.values()]) if stats else np.nan,
            "avg_precision": np.mean([v[1] for v in stats.values()]) if stats else np.nan,
        })
        if len(classes) == 2:
            overlay[name] = (df, classes, pos)
        print(f"{name}: {len(df)} images, accuracy {rows[-1]['accuracy']:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "comparison_summary.csv", index=False)
    print("\n", summary.round(4).to_string(index=False))

    if len(overlay) > 1:
        fr, ar = plt.subplots(figsize=(5.6, 5))
        fp, ap_ = plt.subplots(figsize=(5.6, 5))
        for name, (df, classes, pos) in overlay.items():
            _, y, sc = next(curves(df, classes, pos))
            if y.min() == y.max():
                continue
            fpr, tpr, _ = roc_curve(y, sc)
            prec, rec, _ = precision_recall_curve(y, sc)
            ar.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, sc):.4f})")
            ap_.plot(rec, prec, label=f"{name} (AP {average_precision_score(y, sc):.4f})")
        ar.plot([0, 1], [0, 1], "k--", lw=0.8)
        ar.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC by dataset")
        ap_.set(xlabel="Recall", ylabel="Precision", title="Precision-recall by dataset", ylim=(0, 1.02))
        ar.legend(loc="lower right");
        ap_.legend(loc="lower left")
        fr.tight_layout();
        fp.tight_layout()
        fr.savefig(args.out_dir / "comparison_roc.png", dpi=150)
        fp.savefig(args.out_dir / "comparison_pr.png", dpi=150)
        plt.close("all")

    print(f"\nPlots and tables written to {args.out_dir}")


if __name__ == "__main__":
    main()
