#!/usr/bin/env python3
"""Stage 3: train and evaluate classifiers with stratified cross-validation.

Models (in order of complexity):
  1. Logistic regression (L2, class-balanced) — interpretable baseline
  2. Random forest (class-balanced) — non-linear baseline
  3. XGBoost (skipped gracefully if xgboost is not installed)

Metrics per model (mean +/- std over 5 folds): accuracy, precision, recall,
F1, ROC-AUC. Also saves out-of-fold predictions and top feature importances.

Outputs: results/metrics.json, results/predictions.tsv,
results/feature_importance.tsv.

Checkpoint: exits 0 if results/metrics.json exists (pass --force to re-run).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent.parent
X_PATH = ROOT / "data" / "X.npz"
PHENO = ROOT / "data" / "phenotypes.tsv"
GENES = ROOT / "data" / "feature_names.tsv"
METRICS_OUT = ROOT / "results" / "metrics.json"
PRED_OUT = ROOT / "results" / "predictions.tsv"
IMP_OUT = ROOT / "results" / "feature_importance.tsv"
SEED = 42
N_SPLITS = 5


def build_models():
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
    }
    if HAS_XGB:
        # scale_pos_weight handles the R/S imbalance explicitly.
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=SEED)
    else:
        print("NOTE: xgboost not installed; skipping XGBoost "
              "(conda install -c conda-forge xgboost to enable).")
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if METRICS_OUT.exists() and not args.force:
        print(f"CHECKPOINT PASS: {METRICS_OUT} already exists. "
              "Use --force to re-train.")
        return
    for p in (X_PATH, PHENO, GENES):
        if not p.exists():
            print(f"FAIL: {p} not found. Run stages 1-2 first.", file=sys.stderr)
            sys.exit(1)

    ROOT.joinpath("results").mkdir(parents=True, exist_ok=True)

    X = sparse.load_npz(X_PATH)
    y = (pd.read_csv(PHENO, sep="\t")["label"] == "R").astype(int).values
    genome_ids = pd.read_csv(PHENO, sep="\t")["genome_id"].tolist()
    gene_names = pd.read_csv(GENES, sep="\t")["gene"].tolist()
    print(f"Stage 3: {X.shape[0]} genomes, {X.shape[1]} genes, "
          f"{y.sum()} R / {(1 - y).sum()} S.")

    # XGBoost gets the imbalance ratio; sklearn models use class_weight.
    n_pos, n_neg = y.sum(), (1 - y).sum()
    models = build_models()
    if HAS_XGB and n_pos > 0:
        models["xgboost"].set_params(scale_pos_weight=n_neg / n_pos)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    metrics, pred_frames, importances = {}, {}, []

    for name, model in models.items():
        print(f"Training {name} ({N_SPLITS}-fold CV)...")
        fold_scores = {"accuracy": [], "precision": [], "recall": [],
                       "f1": [], "roc_auc": []}
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            fold_scores["accuracy"].append(accuracy_score(y[test_idx], pred))
            fold_scores["precision"].append(
                precision_score(y[test_idx], pred, zero_division=0))
            fold_scores["recall"].append(recall_score(y[test_idx], pred))
            fold_scores["f1"].append(f1_score(y[test_idx], pred))
            fold_scores["roc_auc"].append(roc_auc_score(y[test_idx], proba))
        metrics[name] = {
            m: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for m, v in fold_scores.items()
        }
        # Out-of-fold predictions for the confusion matrix figure.
        oof_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
        oof_proba = cross_val_predict(model, X, y, cv=cv, n_jobs=-1,
                                      method="predict_proba")[:, 1]
        pred_frames[name] = (oof_pred, oof_proba)
        cm = confusion_matrix(y, oof_pred).tolist()
        metrics[name]["confusion_matrix"] = cm
        auc = metrics[name]["roc_auc"]
        print(f"  {name}: ROC-AUC {auc['mean']:.3f} +/- {auc['std']:.3f}, "
              f"F1 {metrics[name]['f1']['mean']:.3f}")

        # Feature importances (top 100) from the model fit on all data.
        model.fit(X, y)
        if hasattr(model, "coef_"):
            scores = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            scores = model.feature_importances_
        else:
            scores = None
        if scores is not None:
            top = np.argsort(scores)[::-1][:100]
            for rank, j in enumerate(top, 1):
                importances.append({"model": name, "rank": rank,
                                    "gene": gene_names[j],
                                    "score": float(scores[j])})

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({"genome_id": genome_ids, "y_true": y})
    for name, (p, pr) in pred_frames.items():
        pred_df[f"pred_{name}"] = p
        pred_df[f"proba_{name}"] = pr
    pred_df.to_csv(PRED_OUT, sep="\t", index=False)

    pd.DataFrame(importances).to_csv(IMP_OUT, sep="\t", index=False)

    best = max(metrics, key=lambda m: metrics[m]["roc_auc"]["mean"])
    print(f"CHECKPOINT PASS: wrote {METRICS_OUT} (+ predictions, importances).")
    print(f"Best model by ROC-AUC: {best} "
          f"({metrics[best]['roc_auc']['mean']:.3f}).")
    print("Next: python scripts/04_make_figures.py")


if __name__ == "__main__":
    main()
