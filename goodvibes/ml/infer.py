"""Inference service: loads a checkpoint + feature cache, exposes per-sample predictions and spectra."""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from ml import dataset as D
from ml import features as F
from ml.model import VibrationTransformer


class Inferencer:
    def __init__(self, ckpt_path: str, feat_path: str):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = VibrationTransformer(n_cells=ck["n_cells"], d=ck["d"], depth_pt=ck["depth"], depth_st=ck["depth"])
        self.model.load_state_dict(ck["state"]); self.model.eval()
        self.vocab = [tuple(v) for v in ck["vocab"]]
        self.nx, self.ny = int(ck["nx"]), int(ck["ny"])
        self.grid_rc = np.asarray(ck["grid_rc"])
        self.spec, self.band, _, self.table = D.load_features(feat_path)
        self.diff = bool(ck.get("diff", False))
        self.raw_spec = self.spec
        if self.diff:
            self.spec = D.differential(self.spec, self.table)
        self.mu = np.squeeze(np.asarray(ck.get("mu", 0), np.float32))
        self.sd = np.squeeze(np.asarray(ck.get("sd", 1), np.float32))
        self.lab = D.labels(self.table, self.vocab, self.grid_rc)
        self.by_id = {s.sample_id: i for i, s in enumerate(self.table)}
        emp = [i for i, s in enumerate(self.table) if D.is_empty(s)]
        self.empty_ref = self.spec[emp].mean(0) if emp else self.spec.mean(0)   # [P, 2, N]
        self.splits = D.splits(self.table)

    def samples(self, split: str = "S1", part: str = "test") -> list[dict]:
        tr, te = self.splits[split]
        idx = te if part == "test" else tr
        return [self._row(i) for i in idx]

    def _row(self, i: int) -> dict:
        s = self.table[i]
        return {"sample_id": s.sample_id, "speakers": s.speakers, "object": s.obj, "x": s.x, "y": s.y,
                "experiment_id": s.experiment_id}

    @torch.no_grad()
    def predict_spec(self, spec: np.ndarray) -> dict:
        x = torch.as_tensor(((spec - self.mu) / self.sd)[None], dtype=torch.float32)
        out = self.model(x, torch.as_tensor(self.grid_rc, dtype=torch.long))
        pp = torch.softmax(out["presence"], 1)[0].numpy()
        cp = torch.softmax(out["cell"], 1)[0].numpy()
        cell = int(cp.argmax())
        occ = torch.sigmoid(out["occ"])[0].numpy()
        xy = self.vocab[cell] if self.vocab else (-1, -1)
        pos = out["pos"][0].numpy()
        pos_xy = [float(pos[0] * (self.nx - 1)), float(pos[1] * (self.ny - 1))]
        peak = float(occ.max())
        return {"presence": int(pp.argmax()), "presence_prob": float(pp[1]), "confidence": round(peak, 3),
                "pos_xy": [round(v, 2) for v in pos_xy], "cell_xy": [int(round(pos_xy[0])), int(round(pos_xy[1]))],
                "cell": cell, "occ": occ.round(4).tolist(), "occ_grid": self.occ_grid(occ).round(3).tolist(),
                "nx": self.nx, "ny": self.ny}

    def occ_grid(self, occ: np.ndarray) -> np.ndarray:
        g = np.zeros((F.GRID, F.GRID), np.float32)
        for j, (r, c) in enumerate(self.grid_rc):
            g[r, c] = occ[j]
        return g

    def predict(self, sample_id: int) -> dict:
        i = self.by_id[sample_id]
        pred = self.predict_spec(self.spec[i])
        return {"sample": self._row(i), "pred": pred}

    def overhead_path(self, sample_id: int) -> str | None:
        """Resolve image/<image_dir>/cropped_overhead.png next to the data tree (downloaded separately)."""
        s = self.table[self.by_id[sample_id]]
        if s.overhead_path and os.path.exists(s.overhead_path):
            return s.overhead_path
        man = D.load_manifest(s.dir)
        idir = (man.get("sample") or {}).get("image_dir")
        if not idir:
            return None
        root = os.path.dirname(os.path.dirname(s.dir))          # .../data/train
        for name in ("cropped_overhead.png", "segmented_overhead.png", "raw_overhead.png"):
            for base in (root, os.path.join(root, "data")):
                p = os.path.join(base, "image", idir, name)
                if os.path.exists(p):
                    return p
        return None

    def truth(self, sample_id: int) -> dict:
        i = self.by_id[sample_id]
        s = self.table[i]
        return {"presence": int(self.lab["presence"][i]), "x": s.x, "y": s.y, "nx": self.nx, "ny": self.ny,
                "occ_grid": self.occ_grid(self.lab["occ"][i]).round(3).tolist(),
                "overhead_path": self.overhead_path(sample_id), "mask_path": s.mask_path}

    def point_spectrum(self, sample_id: int, laser_idx: int) -> dict:
        i = self.by_id[sample_id]
        sp = self.spec[i][laser_idx]                 # [2, N]
        ref = self.empty_ref[laser_idx]
        step = 10
        return {"laser_idx": laser_idx, "row": int(self.grid_rc[laser_idx][0]), "col": int(self.grid_rc[laser_idx][1]),
                "freqs": F.FREQS[::step].tolist(),
                "x": sp[0][::step].round(3).tolist(), "y": sp[1][::step].round(3).tolist(),
                "ref_x": ref[0][::step].round(3).tolist(), "ref_y": ref[1][::step].round(3).tolist()}

    def compare_with_empty(self, sample_id: int) -> dict:
        """Per-band energy delta (dB-like, log-mag units x 20/ln10) vs the mean empty-box signature."""
        i = self.by_id[sample_id]
        b = F.band_energy(self.spec[i]) - F.band_energy(self.empty_ref)   # [P, B]
        db = (b * 20 / np.log(10)).round(2)
        top = np.argsort(np.abs(db).ravel())[::-1][:8]
        return {"bands_hz": [f"{lo}-{hi}" for lo, hi in F.BANDS],
                "delta_db_grid_by_band": [self.occ_grid(db[:, k]).round(1).tolist() for k in range(db.shape[1])],
                "strongest": [{"laser_idx": int(t // db.shape[1]), "band_hz": f"{F.BANDS[t % db.shape[1]][0]}-{F.BANDS[t % db.shape[1]][1]}",
                               "delta_db": float(db.ravel()[t])} for t in top]}

    def metrics(self, path: str = "results/metrics.json") -> dict:
        return json.load(open(path)) if os.path.exists(path) else {}

    def predict_upload(self, shifts: np.ndarray, fs_hz: float) -> dict:
        sp = F.normalize(F.spectrum(shifts, fs_hz))
        if sp.shape[0] != self.spec.shape[1]:
            raise ValueError(f"expected {self.spec.shape[1]} laser points, got {sp.shape[0]}")
        return {"pred": self.predict_spec(sp)}
