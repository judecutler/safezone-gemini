"""Logistic-regression floor on band-energy maps. Must be beaten by the transformer."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _fit(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    clf.fit(X, y)
    return clf


def pos_metrics(pred_pos: np.ndarray, true_pos: np.ndarray, nx: int, ny: int) -> dict:
    scale = np.array([nx - 1, ny - 1], np.float32)
    err = np.abs(pred_pos - true_pos) * scale              # grid units
    within1 = np.all(err <= 1.0 + 1e-6, axis=1)
    exact = np.all(np.round(pred_pos * scale) == np.round(true_pos * scale), axis=1)
    return {"pos_mae": float(err.mean()), "within1_acc": float(within1.mean()), "exact_acc": float(exact.mean())}


def run(band: np.ndarray, lab: dict, tr: np.ndarray, te: np.ndarray, vocab_len: int) -> dict:
    X = band.reshape(len(band), -1)
    out = {}
    if len(set(lab["presence"][tr])) > 1:
        p = _fit(X[tr], lab["presence"][tr]).predict(X[te])
        out["presence_acc"] = float((p == lab["presence"][te]).mean())
    trm = tr[lab["presence"][tr] == 1]
    tem = te[lab["presence"][te] == 1]
    if len(trm) > 5 and len(tem):
        reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X[trm], lab["pos"][trm])
        pp = np.clip(reg.predict(X[tem]), 0, 1)
        out.update({"ridge_" + k: v for k, v in pos_metrics(pp, lab["pos"][tem], lab["nx"], lab["ny"]).items()})
        knn = make_pipeline(StandardScaler(), KNeighborsRegressor(5, weights="distance")).fit(X[trm], lab["pos"][trm])
        pp = np.clip(knn.predict(X[tem]), 0, 1)
        out.update({"knn_" + k: v for k, v in pos_metrics(pp, lab["pos"][tem], lab["nx"], lab["ny"]).items()})
        hgb = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)).fit(X[trm], lab["pos"][trm])
        pp = np.clip(hgb.predict(X[tem]), 0, 1)
        out.update({"hgb_" + k: v for k, v in pos_metrics(pp, lab["pos"][tem], lab["nx"], lab["ny"]).items()})
        # headline classical baseline = the strongest of the three
        best = min(("ridge", "knn", "hgb"), key=lambda n: out[n + "_pos_mae"])
        out.update({k: out[best + "_" + k] for k in ("pos_mae", "within1_acc", "exact_acc")})
        out["best_classical"] = best
    return out


def _cell_mae(pred, true, lab) -> float:
    nx = lab["nx"]
    px, py = pred % nx, pred // nx
    tx, ty = true % nx, true // nx
    return float((np.abs(px - tx) + np.abs(py - ty)).mean())
