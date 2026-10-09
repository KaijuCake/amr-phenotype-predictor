#!/usr/bin/env python3
"""Stage 4: publication-style figures from the trained models.

  figures/class_balance.png   - R vs S counts in the labeled cohort
  figures/roc_curves.png      - ROC curves per model (out-of-fold)
  figures/confusion_matrix.png - confusion matrix of the best model
  figures/top_features.png    - top 20 predictive genes of the best model

Reads results/metrics.json, results/predictions.tsv,
results/feature_importance.tsv. No model training here.

Checkpoint: exits 0 if all four figures exist (pass --force to re-run).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
EXPECTED = ["class_balance.png", "roc_curves.png",
            "confusion_matrix.png", "top_features.png"]

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
sns.set_style("whitegrid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    FIGDIR.mkdir(parents=True, exist_ok=True)
    if all((FIGDIR / f).exists() for f in EXPECTED) and not args.force:
        print(f"CHECKPOINT PASS: figures already exist in {FIGDIR}. "
              "Use --force to redraw.")
        return
    for p in (ROOT / "results" / "metrics.json",
              ROOT / "results" / "predictions.tsv",
              ROOT / "results" / "feature_importance.tsv"):
        if not p.exists():
            print(f"FAIL: {p} not found. Run stage 3 first.", file=sys.stderr)
            sys.exit(1)

    metrics = json.load(open(ROOT / "results" / "metrics.json"))
    preds = pd.read_csv(ROOT / "results" / "predictions.tsv", sep="\t")
    imp = pd.read_csv(ROOT / "results" / "feature_importance.tsv", sep="\t")
    y = preds["y_true"].values
    best = max(metrics, key=lambda m: metrics[m]["roc_auc"]["mean"])
    print(f"Stage 4: drawing figures (best model: {best}).")

    # 1. Class balance.
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    counts = pd.Series(y).map({1: "Resistant", 0: "Susceptible"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax,
                palette=["#c0392b", "#2980b9"])
    ax.set_ylabel("Genomes")
    ax.set_title("Cohort class balance (carbapenem R vs S)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGDIR / "class_balance.png")
    plt.close(fig)

    # 2. ROC curves (out-of-fold probabilities).
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name in metrics:
        fpr, tpr, _ = roc_curve(y, preds[f"proba_{name}"])
        auc = metrics[name]["roc_auc"]["mean"]
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — carbapenem R vs S")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "roc_curves.png")
    plt.close(fig)

    # 3. Confusion matrix for the best model.
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y, preds[f"pred_{best}"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred S", "Pred R"],
                yticklabels=["True S", "True R"])
    ax.set_title(f"Confusion matrix — {best} (out-of-fold)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "confusion_matrix.png")
    plt.close(fig)

    # 4. Top predictive genes for the best model.
    top = imp[imp["model"] == best].nsmallest(20, "rank")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    order = top.sort_values("score")
    sns.barplot(x="score", y="gene", data=order, ax=ax, color="#2c3e50")
    ax.set_xlabel("Importance score")
    ax.set_title(f"Top 20 predictive AMR genes — {best}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "top_features.png")
    plt.close(fig)

    print(f"CHECKPOINT PASS: wrote {len(EXPECTED)} figures to {FIGDIR}/.")
    print("Done. Fill the [bracketed] fields in README.md with your numbers,")
    print("then update the resume bullets (see docs/runbook.md stage 5).")


if __name__ == "__main__":
    main()
