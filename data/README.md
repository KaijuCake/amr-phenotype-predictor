# data/

This directory holds downloaded data. Nothing here is committed to git
(see `.gitignore`) — every file is produced by running the scripts.

| File | Produced by | Contents |
|---|---|---|
| `phenotype_records.jsonl` | `01_download_phenotypes.py` | Raw BV-BRC `genome_amr` records (one JSON per line) |
| `phenotypes.tsv` | `01_download_phenotypes.py` | One row per genome: `genome_id`, `genome_name`, `label` (R/S), vote counts |
| `X.npz` | `02_build_features.py` | Sparse CSR feature matrix (genomes × AMR genes) |
| `feature_names.tsv` | `02_build_features.py` | Column labels for `X.npz` |
| `genome_ids.tsv` | `02_build_features.py` | Row labels for `X.npz` (same order as `phenotypes.tsv`) |

Re-running a script with `--force` overwrites its outputs.
