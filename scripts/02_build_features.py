#!/usr/bin/env python3
"""Stage 2: build a gene presence/absence feature matrix.

For each genome in data/phenotypes.tsv, queries the BV-BRC sp_gene collection
for its annotated "Antibiotic Resistance" specialty genes (sources: CARD,
NDARO, PATRIC) and builds a sparse binary matrix: rows = genomes,
columns = AMR genes, value = 1 if the genome carries the gene.

Genome IDs are queried in chunks via RQL in() to keep request counts low.

Outputs: data/X.npz (CSR matrix), data/genome_ids.tsv (rows),
data/feature_names.tsv (columns).

Checkpoint: exits 0 if data/X.npz exists (pass --force to re-run).
"""
import argparse
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from tqdm import tqdm

BASE = "https://www.bv-brc.org/api"
# RQL needs the multi-word property value wrapped in encoded double quotes.
AMR_PROPERTY = quote('"Antibiotic Resistance"', safe="")
CHUNK = 40          # genomes per API request
PAGE_SIZE = 5000    # sp_gene records per request (paginated if exceeded)
SLEEP = 0.2
RETRIES = 5

ROOT = Path(__file__).resolve().parent.parent
PHENO = ROOT / "data" / "phenotypes.tsv"
X_OUT = ROOT / "data" / "X.npz"
ROWS_OUT = ROOT / "data" / "genome_ids.tsv"
COLS_OUT = ROOT / "data" / "feature_names.tsv"


def fetch_chunk(genome_ids, offset=0):
    id_list = ",".join(genome_ids)
    q = (
        f"in(genome_id,({id_list}))"
        f"&eq(property,{AMR_PROPERTY})"
        "&select(genome_id,source_id,gene,product,source)"
        f"&limit({PAGE_SIZE},{offset})"
    )
    url = f"{BASE}/sp_gene/?{q}"
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            wait = 2 ** attempt
            print(f"  request failed ({e}); retrying in {wait}s...")
            time.sleep(wait)
    print("FAIL: API request failed after retries.", file=sys.stderr)
    sys.exit(1)


def gene_key(rec):
    """Stable identifier for one AMR gene annotation."""
    sid = (rec.get("source_id") or "").strip()
    if sid:
        return sid
    gene = (rec.get("gene") or "").strip()
    src = (rec.get("source") or "").strip()
    if gene:
        return f"{src}|{gene}"
    return f"{src}|{(rec.get('product') or 'unknown').strip()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if X_OUT.exists() and not args.force:
        print(f"CHECKPOINT PASS: {X_OUT} already exists. Use --force to rebuild.")
        return
    if not PHENO.exists():
        print(f"FAIL: {PHENO} not found. Run scripts/01_download_phenotypes.py first.",
              file=sys.stderr)
        sys.exit(1)

    pheno = pd.read_csv(PHENO, sep="\t")
    genome_ids = pheno["genome_id"].tolist()
    print(f"Stage 2: fetching AMR gene annotations for {len(genome_ids)} genomes...")

    # Pass 1: collect the gene vocabulary.
    vocab = {}       # gene_key -> column index
    genome_genes = {}  # genome_id -> set of gene_keys
    chunks = [genome_ids[i:i + CHUNK] for i in range(0, len(genome_ids), CHUNK)]
    for chunk in tqdm(chunks, desc="gene vocab", unit="chunk"):
        offset = 0
        while True:
            recs = fetch_chunk(chunk, offset)
            for rec in recs:
                gid = rec.get("genome_id")
                key = gene_key(rec)
                if key not in vocab:
                    vocab[key] = len(vocab)
                genome_genes.setdefault(gid, set()).add(key)
            if len(recs) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        time.sleep(SLEEP)

    print(f"Vocabulary: {len(vocab)} distinct AMR genes.")
    missing = [g for g in genome_ids if g not in genome_genes]
    if missing:
        print(f"WARNING: {len(missing)} genomes returned no AMR annotations "
              f"(kept as all-zero rows).")

    # Pass 2: assemble the sparse matrix in phenotype order.
    n, m = len(genome_ids), len(vocab)
    mat = sparse.lil_matrix((n, m), dtype=np.uint8)
    for i, gid in enumerate(genome_ids):
        for key in genome_genes.get(gid, ()):
            mat[i, vocab[key]] = 1
    X = mat.tocsr()
    print(f"Matrix: {X.shape[0]} genomes x {X.shape[1]} genes, "
          f"{X.nnz} nonzeros (density {X.nnz / (n * m):.4f}).")

    sparse.save_npz(X_OUT, X)
    pd.DataFrame({"genome_id": genome_ids}).to_csv(ROWS_OUT, sep="\t", index=False)
    inv_vocab = sorted(vocab, key=vocab.get)
    pd.DataFrame({"gene": inv_vocab}).to_csv(COLS_OUT, sep="\t", index=False)
    print(f"CHECKPOINT PASS: wrote {X_OUT}, {ROWS_OUT}, {COLS_OUT}.")
    print("Next: python scripts/03_train_evaluate.py")


if __name__ == "__main__":
    main()
