"""Train the Vibration Transformer on one split; writes results/best_<split>.pt and metrics."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ml import dataset as D
from ml.model import VibrationTransformer, loss_fn, augment
from ml import baselines


def _to_t(a, dev, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype, device=dev)


def evaluate(model, spec, lab, grid_rc, idx, dev, bs=32) -> tuple[dict, dict]:
    model.eval()
    preds = {"presence": [], "cell": [], "pos": [], "occ": [], "cell_prob": [], "presence_prob": []}
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            out = model(_to_t(spec[b], dev), grid_rc)
            preds["presence"].append(out["presence"].argmax(1).cpu().numpy())
            preds["presence_prob"].append(torch.softmax(out["presence"], 1)[:, 1].cpu().numpy())
            preds["cell"].append(out["cell"].argmax(1).cpu().numpy())
            preds["cell_prob"].append(torch.softmax(out["cell"], 1).cpu().numpy())
            preds["pos"].append(out["pos"].cpu().numpy())
            preds["occ"].append(torch.sigmoid(out["occ"]).cpu().numpy())
    P = {k: np.concatenate(v) for k, v in preds.items()}
    m = {}
    m["presence_acc"] = float((P["presence"] == lab["presence"][idx]).mean())
    obj = lab["presence"][idx] == 1
    if obj.any():
        m.update(baselines.pos_metrics(P["pos"][obj], lab["pos"][idx][obj], lab["nx"], lab["ny"]))
        pred_occ = P["occ"][obj] > 0.5
        true_occ = lab["occ"][idx][obj] > 0.5
        inter = (pred_occ & true_occ).sum(1)
        union = (pred_occ | true_occ).sum(1)
        m["occ_iou"] = float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0)))
    return m, P


def train(split: str, feat_path: str, out_dir: str, epochs: int, bs: int, lr: float, d: int, depth: int, seed: int = 0, aug: bool = True, diff: bool = True):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    spec, band, grid_rc_np, table = D.load_features(feat_path)
    if diff:
        from ml import features as F
        spec = D.differential(spec, table)
        band = np.stack([F.band_energy(x) for x in spec])
    vocab = D.cell_vocab(table)
    lab = D.labels(table, vocab, grid_rc_np)
    tr, te = D.splits(table, seed)[split]
    # small val carve-out from train for early stopping
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr)
    va, tr = perm[: max(4, len(perm) // 10)], perm[max(4, len(perm) // 10):]
    grid_rc = _to_t(grid_rc_np, dev, torch.long)
    # dataset-level per-feature standardization (exposes small per-point spatial contrast)
    mu = spec[tr].mean(0, keepdims=True); sd = spec[tr].std(0, keepdims=True) + 1e-3
    spec = ((spec - mu) / sd).astype(np.float32)

    base = baselines.run(band, lab, np.concatenate([tr, va]), te, len(vocab))
    print(f"[{split}] baseline LR: {base}")

    model = VibrationTransformer(n_cells=len(vocab), d=d, depth_pt=depth, depth_st=depth).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(tr) // bs), pct_start=0.1)
    batch_t = {k: _to_t(v, dev, torch.long if k in ("presence", "cell") else torch.float32)
               for k, v in lab.items() if isinstance(v, np.ndarray)}
    best, best_state, t0 = 1e9, None, time.time()
    for ep in range(epochs):
        model.train()
        rng.shuffle(tr)
        tot = 0.0
        for i in range(0, len(tr) - bs + 1, bs):
            b = tr[i:i + bs]
            x = _to_t(spec[b], dev)
            if aug:
                x = augment(x)
            out = model(x, grid_rc)
            bt = {k: v[b] for k, v in batch_t.items()}
            loss, _ = loss_fn(out, bt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        # val loss
        model.eval()
        with torch.no_grad():
            out = model(_to_t(spec[va], dev), grid_rc)
            vl, _ = loss_fn(out, {k: v[va] for k, v in batch_t.items()})
        if vl.item() < best:
            best, best_state = vl.item(), {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[{split}] ep {ep:4d} train {tot / max(1, len(tr) // bs):.3f} val {vl.item():.3f} best {best:.3f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    metrics, preds = evaluate(model, spec, lab, grid_rc, te, dev)
    print(f"[{split}] transformer ({n_par:.1f}M params): {metrics}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state": best_state, "n_cells": len(vocab), "vocab": vocab, "d": d, "depth": depth, "diff": diff,
                "mu": mu.astype(np.float16), "sd": sd.astype(np.float16),
                "grid_rc": grid_rc_np, "nx": lab["nx"], "ny": lab["ny"]}, os.path.join(out_dir, f"best_{split}.pt"))
    # per-sample preds for the app
    pdir = os.path.join(out_dir, "preds")
    os.makedirs(pdir, exist_ok=True)
    for j, i in enumerate(te):
        s = table[i]
        json.dump({
            "sample_id": s.sample_id, "split": split, "speakers": s.speakers,
            "truth": {"presence": int(lab["presence"][i]), "x": s.x, "y": s.y, "occ": lab["occ"][i].round(2).tolist()},
            "pred": {"presence": int(preds["presence"][j]), "presence_prob": float(preds["presence_prob"][j]),
                     "pos_xy": [float(preds["pos"][j][0] * (lab["nx"] - 1)), float(preds["pos"][j][1] * (lab["ny"] - 1))],
                     "occ": preds["occ"][j].round(3).tolist()},
        }, open(os.path.join(pdir, f"{split}_{s.sample_id:07d}.json"), "w"))
    mpath = os.path.join(out_dir, "metrics.json")
    allm = json.load(open(mpath)) if os.path.exists(mpath) else {}
    allm[split] = {"baseline_lr": base, "vibration_transformer": metrics,
                   "n_train": int(len(tr) + len(va)), "n_test": int(len(te)), "params_M": round(n_par, 2)}
    json.dump(allm, open(mpath, "w"), indent=1)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="S1", choices=["S1", "S2", "S3", "all"])
    ap.add_argument("--features", default="results/features.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--no-diff", action="store_true")
    a = ap.parse_args()
    for sp in (["S1", "S2", "S3"] if a.split == "all" else [a.split]):
        train(sp, a.features, a.out, a.epochs, a.bs, a.lr, a.d, a.depth, aug=not a.no_aug, diff=not a.no_diff)
