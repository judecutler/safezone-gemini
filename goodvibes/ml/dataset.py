"""Load the HF `laser-vibrations` tree (or ml/synth.py output) into a sample table + feature cache."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from ml import features as F


@dataclass
class Sample:
    sample_id: int
    experiment_id: str
    obj: str                 # "cube" | "empty" | ...
    n_objects: int
    x: int
    y: int
    x_com: float
    y_com: float
    speakers: str            # 4-bit code
    dir: str                 # per-sample directory
    mask_path: str | None
    overhead_path: str | None


def _find_root(root: str) -> str:
    """Return the directory that holds metadata.jsonl (handles an experiment-16/ prefix)."""
    hits = glob.glob(os.path.join(root, "**", "metadata.jsonl"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no metadata.jsonl under {root}")
    return os.path.dirname(sorted(hits, key=len)[0])


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_table(root: str) -> list[Sample]:
    data_dir = _find_root(root)
    base = os.path.dirname(data_dir) if os.path.basename(data_dir) == "data" else data_dir
    rows = []
    with open(os.path.join(data_dir, "metadata.jsonl")) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    out: list[Sample] = []
    for r in rows:
        man = r.get("manifest")
        if isinstance(man, str):
            try:
                man = json.loads(man)
            except json.JSONDecodeError:
                man = {}
        man = man or {}
        s = man.get("sample", {})
        sid = int(r["sample_id"])
        sdir = os.path.join(data_dir, f"{sid:07d}")
        if not os.path.isdir(sdir):
            cands = glob.glob(os.path.join(base, "**", f"{sid:07d}"), recursive=True)
            sdir = cands[0] if cands else sdir
        mask = _first(r, "mask_file_name")
        over = _first(r, "segmented_overhead_file_name")
        mask_p = _resolve(base, mask)
        over_p = _resolve(base, over)
        out.append(Sample(
            sample_id=sid,
            experiment_id=str(_first(r, "experiment_id", default=s.get("experiment_id", ""))),
            obj=str(_first(r, "object", default=s.get("object", "unknown"))),
            n_objects=int(_first(r, "n_objects", default=s.get("n_objects", 0)) or 0),
            x=int(_first(r, "x_position", default=s.get("x_position", -1))),
            y=int(_first(r, "y_position", default=s.get("y_position", -1))),
            x_com=float(_first(r, "x_com", default=-1.0) or -1.0),
            y_com=float(_first(r, "y_com", default=-1.0) or -1.0),
            speakers=str(_first(r, "speakers", default=s.get("speakers", "0000"))),
            dir=sdir,
            mask_path=mask_p,
            overhead_path=over_p,
        ))
    return out


def _resolve(base: str, rel) -> str | None:
    if not rel or not isinstance(rel, str):
        return None
    for cand in (os.path.join(base, rel), os.path.join(base, "data", rel)):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(base, "**", os.path.basename(rel)), recursive=True)
    return hits[0] if hits else None


def load_manifest(sdir: str) -> dict:
    p = os.path.join(sdir, "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def load_shifts(sdir: str) -> tuple[np.ndarray, float, dict]:
    """Return ([n_lasers, T, 2], fs_hz, manifest). Prefers the cleaned shifts."""
    man = load_manifest(sdir)
    fs = float(_dig(man, "processing_config", "speckle_shifts", "fs_hz", default=0) or 0)
    for name in ("speckle_shifts_clean.npz", "speckle_shifts.npz"):
        p = os.path.join(sdir, name)
        if os.path.exists(p):
            z = np.load(p)
            arrs = [z[k] for k in z.files if getattr(z[k], "ndim", 0) == 3]
            if not arrs:
                continue
            if fs <= 0:
                for k in z.files:
                    if "fs" in k.lower() and z[k].size == 1:
                        fs = float(z[k])
            if fs <= 0:
                raise ValueError(f"fs_hz missing for {sdir}; set processing_config.speckle_shifts.fs_hz")
            return F.canonical_shifts(arrs[0]), fs, man
    raise FileNotFoundError(f"no shifts npz in {sdir}")


def _dig(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def is_empty(s: Sample) -> bool:
    return s.obj.lower() == "empty" or s.n_objects == 0


def build_features(root: str, out_path: str) -> dict:
    """Compute and cache features for every sample. Returns the cache dict."""
    table = load_table(root)
    specs, bands, rc = [], [], None
    for s in table:
        shifts, fs, man = load_shifts(s.dir)
        sp = F.normalize(F.spectrum(shifts, fs))
        specs.append(sp.astype(np.float16))
        bands.append(F.band_energy(sp))
        if rc is None:
            rois = _dig(man, "experiment_config", "laser_grid", "sensor_rois_xywh")
            rc = F.grid_order(shifts.shape[0], rois)
    cache = dict(
        spec=np.stack(specs), band=np.stack(bands), grid_rc=rc,
        meta=json.dumps([asdict(s) for s in table]),
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **cache)
    return cache


def load_features(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Sample]]:
    z = np.load(path, allow_pickle=False)
    meta = [Sample(**m) for m in json.loads(str(z["meta"]))]
    return z["spec"].astype(np.float32), z["band"], z["grid_rc"], meta


def differential(spec: np.ndarray, table: list[Sample]) -> np.ndarray:
    """Differential measurement: subtract the empty-container reference recorded with the same excitation
    (one empty box per speaker code in this dataset). Falls back to the same-speaker mean if no empty exists."""
    spk = np.array([s.speakers for s in table])
    out = spec.copy()
    for code in np.unique(spk):
        m = spk == code
        emp = [i for i in np.where(m)[0] if is_empty(table[i])]
        ref = spec[emp].mean(0) if emp else spec[m].mean(0)
        out[m] = spec[m] - ref
    return out


# ---------------- labels & splits ----------------

def cell_vocab(table: list[Sample]) -> list[tuple[int, int]]:
    cells = sorted({(s.x, s.y) for s in table if not is_empty(s) and s.x >= 0 and s.y >= 0})
    return cells


def labels(table: list[Sample], vocab: list[tuple[int, int]], grid_rc: np.ndarray, sigma: float = 1.0) -> dict:
    """presence, cell index, normalized pos=(x/(nx-1), y/(ny-1)), gaussian occupancy bump on the laser grid."""
    idx = {c: i for i, c in enumerate(vocab)}
    xs = [c[0] for c in vocab] or [0]
    ys = [c[1] for c in vocab] or [0]
    nx, ny = max(xs) + 1, max(ys) + 1
    n = len(table)
    presence = np.zeros(n, np.int64)
    cell = np.full(n, -1, np.int64)
    pos = np.zeros((n, 2), np.float32)
    occ = np.zeros((n, grid_rc.shape[0]), np.float32)
    # laser (row, col) -> position-grid coordinates
    lr = grid_rc[:, 0] * (ny - 1) / max(F.GRID - 1, 1)
    lc = grid_rc[:, 1] * (nx - 1) / max(F.GRID - 1, 1)
    for i, s in enumerate(table):
        if is_empty(s):
            continue
        presence[i] = 1
        cell[i] = idx.get((s.x, s.y), -1)
        pos[i] = [s.x / max(nx - 1, 1), s.y / max(ny - 1, 1)]
        d2 = (lc - s.x) ** 2 + (lr - s.y) ** 2
        occ[i] = np.exp(-d2 / (2 * sigma ** 2))
    return dict(presence=presence, cell=cell, pos=pos, com=pos.copy(), occ=occ, nx=nx, ny=ny)


def splits(table: list[Sample], seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = len(table)
    idx = np.arange(n)
    spk = np.array([s.speakers for s in table])
    codes = sorted(set(spk))
    # S1: unseen excitation (hold out the last speaker code)
    held = codes[-1] if len(codes) > 1 else None
    s1_te = idx[spk == held] if held else idx[rng.random(n) < 0.2]
    s1_tr = np.setdiff1d(idx, s1_te)
    # S2: unseen location (hold out ~20% of cells; empties split randomly)
    cells = cell_vocab(table)
    held_cells = set(map(tuple, rng.permutation(cells)[: max(1, len(cells) // 5)])) if cells else set()
    s2_te = np.array([i for i, s in enumerate(table)
                      if (not is_empty(s) and (s.x, s.y) in held_cells)
                      or (is_empty(s) and rng.random() < 0.2)], dtype=np.int64)
    s2_tr = np.setdiff1d(idx, s2_te)
    # S3: random
    perm = rng.permutation(n)
    s3_te, s3_tr = perm[: n // 5], perm[n // 5:]
    return {"S1": (s1_tr, s1_te), "S2": (s2_tr, s2_te), "S3": (s3_tr, s3_te)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/train")
    ap.add_argument("--out", default="results/features.npz")
    a = ap.parse_args()
    t = load_table(a.root)
    from collections import Counter
    print(f"{len(t)} samples; objects={Counter(s.obj for s in t)}; speakers={Counter(s.speakers for s in t)}")
    print("cells:", Counter((s.x, s.y) for s in t if not is_empty(s)))
    c = build_features(a.root, a.out)
    print("spec", c["spec"].shape, "band", c["band"].shape, "->", a.out)
