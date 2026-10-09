# Runbook — AMR Phenotype Predictor

Predicting carbapenem resistance (R vs S) in *Acinetobacter baumannii* from
annotated AMR genes, using the public BV-BRC database. Runs on a laptop;
no wet lab, no HPC, no API key.

Each stage ends with a **pass/fail checkpoint**. If a stage fails, fix it
before moving on — later stages depend on its outputs.

---

## Stage 0 — Environment setup

```bash
cd ~/amr-phenotype-predictor
conda env create -f environment.yml
conda activate amr-predictor
```

- **Pass:** `python -c "import sklearn, pandas, scipy, requests; print('ok')"` prints `ok`.
- **Fail:** conda solver errors → try `conda env create -f environment.yml --solver libmamba`
  (or `mamba env create -f environment.yml`).
- XGBoost is optional: stage 3 skips it gracefully if the install fails.
  To add it later: `conda install -c conda-forge xgboost`, then re-run stage 3
  with `--force`.

Expected time: 5–15 min (one-time download of packages).

---

## Stage 1 — Download susceptibility labels

```bash
python scripts/01_download_phenotypes.py
```

What it does: pages the BV-BRC `genome_amr` API for *A. baumannii*
(taxon 470) tested against meropenem / imipenem / doripenem / ertapenem
(verified Oct 2026: 64,378 records — 21,127 R / 12,133 S / 144 I), then
assigns each genome one label by majority vote. Ties and
intermediate-only genomes are dropped. Cohorts larger than 4,000 genomes
are stratified-downsampled (seed 42) to stay laptop-friendly; adjust with
`--max-genomes`.

- **Pass:** ends with `CHECKPOINT PASS: wrote data/phenotypes.tsv (N genomes).`
  Sanity-check: `head data/phenotypes.tsv`, and confirm both R and S labels
  are well represented (expect roughly 60/40 R/S before the cap).
- **Fail:** API timeouts → the script retries with backoff; if your network
  blocks bv-brc.org, check VPN/firewall. Re-running skips nothing until the
  TSV is written; use `--force` to start over.

Expected time: 10–30 min (polite 0.2 s delay between ~65 API pages).

---

## Stage 2 — Build the gene presence/absence matrix

```bash
python scripts/02_build_features.py
```

What it does: for each labeled genome, pulls its annotated
"Antibiotic Resistance" specialty genes from the BV-BRC `sp_gene` collection
(sources: CARD, NDARO, PATRIC — ~100 genes per genome) in chunks of 40
genomes, then assembles a sparse binary matrix (genomes × genes).

- **Pass:** ends with `CHECKPOINT PASS: wrote data/X.npz ...`
  Sanity-check: matrix shape printed, e.g. `4000 genomes x ~1500 genes`,
  density well under 0.1.
- **Fail:** `WARNING: N genomes returned no AMR annotations` — a handful is
  fine (kept as all-zero rows); if it's hundreds, the API may be throttling —
  wait 10 min and re-run (completed chunks are not re-fetched... actually
  they are; the script is idempotent but not resumable mid-run, so a failure
  restarts the fetch — each chunk is fast, total ~15–40 min).

Expected time: 15–45 min depending on cohort size.

---

## Stage 3 — Train and evaluate

```bash
python scripts/03_train_evaluate.py
```

What it does: stratified 5-fold cross-validation over three models —
logistic regression → random forest → XGBoost — reporting accuracy,
precision, recall, F1, ROC-AUC (mean ± std), out-of-fold predictions, and
top-100 feature importances per model.

- **Pass:** ends with `CHECKPOINT PASS` and prints each model's ROC-AUC.
  Sanity-check: models should beat chance (AUC > 0.5) and the stronger
  models should beat logistic regression. If AUC is ~0.5 for everything,
  the labels didn't align with the features — stop and debug (check
  `data/phenotypes.tsv` row order vs `data/genome_ids.tsv`) before
  believing any figure.
- **Fail:** out-of-memory — unlikely (sparse matrix, a few thousand rows);
  if it happens, lower `--max-genomes` in stage 1 and re-run stages 2–3
  with `--force`.

Expected time: 2–10 min.

---

## Stage 4 — Figures

```bash
python scripts/04_make_figures.py
```

What it does: draws `figures/class_balance.png`, `figures/roc_curves.png`,
`figures/confusion_matrix.png` (best model), `figures/top_features.png`
(top 20 genes).

- **Pass:** `CHECKPOINT PASS: wrote 4 figures`. Open them and eyeball:
  ROC curves above the diagonal, confusion matrix not degenerate
  (both classes predicted), top genes biologically plausible
  (e.g. *bla*OXA carbapenemases, efflux pumps).
- **Fail:** missing `results/` files → re-run stage 3.

---

## Stage 5 — Write it up

1. Fill every `[bracketed]` field in `README.md` with your real numbers
   (cohort size, per-model AUC/F1, top genes + one-line interpretation).
2. Skim `results/feature_importance.tsv` — if the top hits are known
   carbapenem-resistance genes, say so in the README; that's the story.
3. Update the resume: replace the design-only AMR bullet with claimed
   results, e.g. *"Trained R-vs-S classifiers on [N] A. baumannii genomes
   (BV-BRC); best model [XGBoost/RF] reached ROC-AUC [X.XX]; top features
   recovered known carbapenemases."*
4. `git init`, commit, push to GitHub; link the repo on the resume.

---

## Troubleshooting quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `requests` timeouts in stage 1/2 | BV-BRC throttling or flaky network | Wait, re-run (retries built in) |
| `sp_gene` returns `[]` for a genome | Genome has no AMR annotations | Expected for a few; kept as zero rows |
| XGBoost missing | conda install hiccup | Stage 3 skips it; install later + `--force` |
| AUC ≈ 0.5 everywhere | Label/feature misalignment | Verify row order matches between stages |
| Conda solver hangs | Classic solver on large env | Use libmamba/mamba |

## Extending to the other two datasets

The scripts are organism-agnostic apart from constants at the top of
`01_download_phenotypes.py`: change `TAXON_ID` (1280 = *S. aureus*,
562 = *E. coli*) and `ANTIBIOTICS` (e.g. `["oxacillin", "cefoxitin"]` for
MRSA, `["ciprofloxacin"]` for *E. coli*), then re-run stages 1–4 with
`--force` in a fresh clone. Keep one repo per dataset for clean GitHub
presentation.
