"""Gemini Embedding 2 on rendered signatures -> cosine kNN. Zero training, pure Google model."""
from __future__ import annotations

import numpy as np

from gemini.client import client, unavailable, MODEL_EMBED
from gemini.render import signature_png


def embed_png(png: bytes, model: str = MODEL_EMBED) -> np.ndarray | None:
    c = client()
    if c is None:
        return None
    from google.genai import types
    # NOTE: verify the multimodal embed_content call shape against current docs once the key is available.
    resp = c.models.embed_content(model=model, contents=[types.Part.from_bytes(data=png, mime_type="image/png")])
    return np.asarray(resp.embeddings[0].values, dtype=np.float32)


def evaluate(spec, lab, table, grid_rc, train_idx, test_idx, k: int = 5, cache: dict | None = None) -> dict:
    if client() is None:
        return unavailable()
    cache = cache if cache is not None else {}

    def emb(i):
        if i not in cache:
            try:
                cache[i] = embed_png(signature_png(spec[i], grid_rc))
            except Exception as e:
                cache[i] = e
        return cache[i]

    E_tr = []
    for i in train_idx:
        e = emb(i)
        if isinstance(e, Exception):
            return unavailable(f"{type(e).__name__}: {e}")
        E_tr.append(e)
    E_tr = np.stack(E_tr)
    E_tr /= np.linalg.norm(E_tr, axis=1, keepdims=True) + 1e-9
    errs = []
    scale = np.array([lab["nx"] - 1, lab["ny"] - 1], np.float32)
    for i in test_idx:
        e = emb(i)
        if isinstance(e, Exception):
            return unavailable(f"{type(e).__name__}: {e}")
        e = e / (np.linalg.norm(e) + 1e-9)
        nn = train_idx[np.argsort(-(E_tr @ e))[:k]]
        nn = [j for j in nn if lab["presence"][j] == 1]
        if lab["presence"][i] == 1 and nn:
            pred = lab["pos"][nn].mean(0)
            errs.append(np.abs(pred - lab["pos"][i]) * scale)
    e = np.asarray(errs)
    return {"available": True, "pos_mae": float(e.mean()), "within1_acc": float(np.all(e <= 1, axis=1).mean()),
            "exact_acc": float(np.all(np.round(e) == 0, axis=1).mean()), "n": len(test_idx), "k": k, "model": MODEL_EMBED}
