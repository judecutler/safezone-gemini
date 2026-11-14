"""Score the Gemini zero-training paths on the held-out splits -> results/metrics_gemini.json (+ few-shot cache).

Usage: GEMINI_API_KEY=... python -m gemini.evaluate --features results/features.npz --splits S1 --limit 40
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ml import dataset as D
from gemini import listener, embed_knn


def main(a):
    spec, band, grid_rc, table = D.load_features(a.features)
    vocab = D.cell_vocab(table)
    lab = D.labels(table, vocab, grid_rc)
    sp = D.splits(table)
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for s in a.splits:
        tr, te = sp[s]
        res = {}
        if "fewshot" in a.which:
            res["fewshot"] = listener.evaluate(spec, lab, table, grid_rc, tr, te, lab["nx"], lab["ny"], k=a.k, limit=a.limit)
            print(s, "fewshot:", {k: v for k, v in res["fewshot"].items() if k != "rows"})
        if "knn" in a.which:
            res["knn"] = embed_knn.evaluate(spec, lab, table, grid_rc, tr, te[: a.limit] if a.limit else te)
            print(s, "knn:", res["knn"])
        out[s] = res
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/features.npz")
    ap.add_argument("--out", default="results/metrics_gemini.json")
    ap.add_argument("--splits", nargs="+", default=["S1"])
    ap.add_argument("--which", nargs="+", default=["fewshot", "knn"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args())
