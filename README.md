# Predicting Antibiotic Resistance from Bacterial DNA

*Can we look at a bacterium's genome and predict whether antibiotics will
kill it — without running the lab test?*

> **Status:** complete — the pipeline ran end-to-end on 2026-10-09 and every
> number below is a real measured result.

---

## Why this matters

Antibiotics are one of medicine's greatest inventions — and they are slowly
failing. Bacteria evolve resistance, and resistant infections are harder,
slower, and more expensive to treat.

Today, figuring out whether a bacterium resists a given antibiotic means
**growing it in the lab and exposing it to the drug** — reliable, but it
takes days, during which a patient may be on the wrong treatment.

This project asks a modern question: **the bacterium's DNA already contains
the instructions for its resistance machinery** (pumps that eject drugs,
enzymes that chew them up). If we can read those instructions, can a computer
learn to predict the lab result directly from the genome — in minutes instead
of days?

That is exactly what this repository does, for one of the most urgent cases
in medicine: *Acinetobacter baumannii* (a hospital pathogen) versus
carbapenems (a last-resort antibiotic family). Carbapenem-resistant
*A. baumannii* sits on the WHO's critical-priority list.

## What this project does, step by step

1. **Collects real data.** Downloads thousands of bacterial genomes together
   with their actual lab-measured drug responses from BV-BRC, a free public
   database run for the research community (formerly PATRIC). No data is
   made up; every label comes from a published laboratory measurement.

2. **Cleans up the labels.** One genome can have several lab records that
   don't perfectly agree, so each genome gets a single verdict —
   *Resistant* or *Susceptible* — by majority vote. Ambiguous cases are
   thrown out rather than guessed.

3. **Turns DNA into a checklist.** For each genome, the project lists which
   known antibiotic-resistance genes it carries (using curated references:
   CARD, NDARO, and PATRIC). Each genome becomes a row of yes/no answers —
   about 2,510 genes checked per genome.

4. **Trains predictors.** Three machine-learning models, from simple to
   sophisticated — logistic regression, random forest, XGBoost — each try to
   predict Resistant vs Susceptible from the checklist.

5. **Grades honestly.** The models are tested with 5-fold cross-validation
   (a standard technique where the model is repeatedly tested on data it has
   never seen), and scored on several metrics, not just one. The repository
   also produces figures: how balanced the data is, how each model performs,
   and which genes the best model relied on most.

## Results (from the first full run)

- **Cohort:** 3,999 *A. baumannii* genomes — 2,611 resistant, 1,388 susceptible
  to carbapenems (raw pull: 64,378 lab records → 14,612 genomes labeled
  9,541 resistant / 5,071 susceptible by majority vote; 678 tied or ambiguous
  genomes dropped)
- **Best model:** XGBoost — ROC-AUC 0.992, F1 score 0.979 (5-fold cross-validation)
- **Full comparison:** logistic regression 0.990 → random forest 0.989 →
  XGBoost 0.992
- **Most predictive genes:** OXA-23 and OXA-24 family carbapenemases —
  the known carbapenem-destroying enzymes in *A. baumannii* — so the model
  rediscovered real resistance biology rather than a spurious pattern

## What's inside

| Stage | Script | What it does |
|---|---|---|
| 1 | `scripts/01_download_phenotypes.py` | Downloads lab results from BV-BRC, votes them into one R/S label per genome |
| 2 | `scripts/02_build_features.py` | Fetches each genome's resistance genes, builds the yes/no checklist matrix |
| 3 | `scripts/03_train_evaluate.py` | Trains the three models, cross-validates, saves metrics and predictions |
| 4 | `scripts/04_make_figures.py` | Draws the figures in `figures/` |

Each script has a pass/fail checkpoint: re-running it skips finished work
unless you pass `--force`. Supporting files: `environment.yml` (the exact
software setup, via conda), `docs/runbook.md` (a detailed walkthrough of
every stage, including what to do when something fails), `data/README.md`
(explains every data file).

## For the technically curious

- **Task:** binary classification, carbapenem R vs S, *A. baumannii*
  (BV-BRC taxon 470; drugs: meropenem, imipenem, doripenem, ertapenem).
- **Features:** sparse binary presence/absence over BV-BRC `sp_gene`
  "Antibiotic Resistance" annotations — interpretable by design, not k-mers.
- **Validation:** stratified 5-fold CV, seed 42; metrics are mean ± std
  across folds. Class imbalance handled via `class_weight="balanced"` /
  `scale_pos_weight`.
- **Reproduce it:**
  ```bash
  git clone <this-repo> ~/amr-phenotype-predictor
  cd ~/amr-phenotype-predictor
  conda env create -f environment.yml && conda activate amr-predictor
  python scripts/01_download_phenotypes.py
  python scripts/02_build_features.py
  python scripts/03_train_evaluate.py
  python scripts/04_make_figures.py
  ```


## Data credit

Genome and susceptibility data: **BV-BRC** (Bacterial and Viral
Bioinformatics Resource Center, https://www.bv-brc.org), funded by NIAID.
Resistance gene annotations: CARD, NDARO, and PATRIC, via BV-BRC.

## License

MIT — see `LICENSE`.
