#!/usr/bin/env python3
"""Stage 1: download carbapenem susceptibility records from BV-BRC and build
per-genome R/S labels.

Queries the public BV-BRC genome_amr collection for Acinetobacter baumannii
(taxon 470) tested against carbapenems, pages through all records, then
assigns each genome a single label by majority vote across its records.
Genomes with tied votes or only Intermediate records are dropped.

Output: data/phenotype_records.jsonl (raw) and data/phenotypes.tsv (labels).

Checkpoint: exits 0 without doing anything if data/phenotypes.tsv already
exists (pass --force to re-run).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE = "https://www.bv-brc.org/api"
TAXON_ID = 470  # Acinetobacter baumannii
ANTIBIOTICS = ["meropenem", "imipenem", "doripenem", "ertapenem"]
PAGE_SIZE = 1000
SLEEP = 0.2  # politeness delay between API calls
RETRIES = 5

ROOT = Path(__file__).resolve().parent.parent
RAW_OUT = ROOT / "data" / "phenotype_records.jsonl"
LABEL_OUT = ROOT / "data" / "phenotypes.tsv"


def fetch_page(offset):
    q = (
        f"eq(taxon_id,{TAXON_ID})"
        f"&in(antibiotic,({','.join(ANTIBIOTICS)}))"
        "&select(id,genome_id,genome_name,antibiotic,resistant_phenotype,"
        "measurement,testing_standard)"
        f"&limit({PAGE_SIZE},{offset})"
    )
    url = f"{BASE}/genome_amr/?{q}"
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            wait = 2 ** attempt
            print(f"  request failed ({e}); retrying in {wait}s...")
            time.sleep(wait)
    print("FAIL: API request failed after retries.", file=sys.stderr)
    sys.exit(1)


def phenotype_to_vote(ph):
    if not ph:
        return None
    ph = ph.strip().lower()
    if ph.startswith("resistant"):
        return "R"
    if ph.startswith("susceptible"):
        return "S"
    return None  # Intermediate / unknown -> ignored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    ap.add_argument("--max-genomes", type=int, default=4000,
                    help="cap cohort size (stratified sample) for laptop friendliness")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if LABEL_OUT.exists() and not args.force:
        n = sum(1 for _ in open(LABEL_OUT)) - 1
        print(f"CHECKPOINT PASS: {LABEL_OUT} already exists ({n} genomes).")
        print("Nothing to do. Use --force to re-download.")
        return

    ROOT.joinpath("data").mkdir(parents=True, exist_ok=True)

    print("Stage 1: downloading carbapenem susceptibility records from BV-BRC...")
    records = []
    offset = 0
    with tqdm(desc="API pages", unit="page") as bar:
        while True:
            page = fetch_page(offset)
            records.extend(page)
            bar.update(1)
            bar.set_postfix(records=len(records))
            if len(page) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            time.sleep(SLEEP)

    print(f"Downloaded {len(records)} raw records.")
    with open(RAW_OUT, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Aggregate to per-genome labels by majority vote.
    votes = defaultdict(lambda: {"R": 0, "S": 0, "name": "", "abx": set()})
    for rec in records:
        gid = rec.get("genome_id")
        if not gid:
            continue
        v = phenotype_to_vote(rec.get("resistant_phenotype"))
        if v is None:
            continue
        votes[gid][v] += 1
        votes[gid]["name"] = rec.get("genome_name", "")
        votes[gid]["abx"].add(rec.get("antibiotic", ""))

    rows = []
    n_ties = 0
    for gid, d in votes.items():
        if d["R"] > d["S"]:
            label = "R"
        elif d["S"] > d["R"]:
            label = "S"
        else:
            n_ties += 1
            continue  # tied vote -> ambiguous, drop
        rows.append({
            "genome_id": gid,
            "genome_name": d["name"],
            "label": label,
            "n_resistant": d["R"],
            "n_susceptible": d["S"],
            "antibiotics": ";".join(sorted(d["abx"])),
        })
    df = pd.DataFrame(rows)
    print(f"Labeled genomes: {len(df)} "
          f"({(df.label == 'R').sum()} R / {(df.label == 'S').sum()} S); "
          f"dropped {n_ties} tied/ambiguous.")

    if len(df) > args.max_genomes:
        # Stratified downsample to keep the feature stage laptop-friendly.
        keep = []
        for lab, grp in df.groupby("label"):
            k = min(len(grp), int(args.max_genomes * len(grp) / len(df)))
            keep.append(grp.sample(n=k, random_state=args.seed))
        df = pd.concat(keep).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"Capped cohort at {len(df)} genomes "
              f"({(df.label == 'R').sum()} R / {(df.label == 'S').sum()} S).")

    df.to_csv(LABEL_OUT, sep="\t", index=False)
    print(f"CHECKPOINT PASS: wrote {LABEL_OUT} ({len(df)} genomes).")
    print("Next: python scripts/02_build_features.py")


if __name__ == "__main__":
    main()
