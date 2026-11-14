"""Gemini as a zero-training inspector: few-shot on rendered vibration signatures (PNG), structured output."""
from __future__ import annotations

import json
import time

import numpy as np

from gemini.client import client, unavailable, MODEL_FAST
from gemini.render import signature_png
from gemini.report_schema import ListenerGuess

SYSTEM = (
    "You are a non-destructive-testing analyst. Each image is the vibration signature of the side wall of a sealed "
    "cardboard box measured on a 10x10 laser grid: five band-energy maps (brighter = more surface vibration in that "
    "band) and the mean spectrum per grid row. A hidden object pressed against the wall locally damps the wall and "
    "shifts resonances. Estimate the object's position (x = column 0..nx-1, y = row 0..ny-1, fractional values allowed) "
    "on the position grid given in the prompt; the laser grid spans the same wall, so a damped region near laser "
    "row r / column c maps to y ≈ r·(ny-1)/9, x ≈ c·(nx-1)/9. Answer with JSON."
)


def _part_png(png: bytes):
    from google.genai import types
    return types.Part.from_bytes(data=png, mime_type="image/png")


def few_shot_predict(spec_query: np.ndarray, shots: list[tuple[np.ndarray, dict]], grid_rc: np.ndarray,
                     nx: int, ny: int, model: str = MODEL_FAST) -> dict:
    """shots: list of (spec, {"presence": bool, "x": int, "y": int})."""
    c = client()
    if c is None:
        return unavailable()
    from google.genai import types
    contents = [f"Position grid: nx={nx} columns, ny={ny} rows. Labeled examples follow."]
    for sp, lab in shots:
        contents.append(_part_png(signature_png(sp, grid_rc)))
        contents.append(f"label: {json.dumps(lab)}")
    contents.append("Now the query. Return JSON only.")
    contents.append(_part_png(signature_png(spec_query, grid_rc)))
    t0 = time.time()
    try:
        resp = c.models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM, response_mime_type="application/json",
                response_schema=ListenerGuess, temperature=0.2),
        )
        g = ListenerGuess.model_validate_json(resp.text)
        return {"available": True, "model": model, "latency_s": round(time.time() - t0, 2), **g.model_dump()}
    except Exception as e:  # never fail the demo
        return unavailable(f"{type(e).__name__}: {e}")


def pick_shots(spec: np.ndarray, lab: dict, train_idx: np.ndarray, table, k: int = 10, seed: int = 0):
    """Balanced few-shot set: ~1/3 empty, rest objects spread across cells."""
    rng = np.random.default_rng(seed)
    emp = [i for i in train_idx if lab["presence"][i] == 0]
    obj = [i for i in train_idx if lab["presence"][i] == 1]
    rng.shuffle(emp); rng.shuffle(obj)
    chosen = emp[: max(1, k // 3)] + obj[: k - max(1, k // 3)]
    return [(spec[i], {"presence": bool(lab["presence"][i]), "x": int(table[i].x), "y": int(table[i].y)}) for i in chosen]


def pick_shots_spread(spec, lab, train_idx, table, k: int = 12, seed: int = 0):
    """Shots spread across the position grid (corners, edges, centre) so Gemini sees the spatial mapping."""
    rng = np.random.default_rng(seed)
    obj = [i for i in train_idx if lab["presence"][i] == 1]
    rng.shuffle(obj)
    chosen, seen = [], set()
    for i in obj:
        key = (table[i].x // 3, table[i].y // 3)
        if key not in seen:
            seen.add(key); chosen.append(i)
        if len(chosen) >= k:
            break
    return [(spec[i], {"presence": True, "x": int(table[i].x), "y": int(table[i].y)}) for i in chosen]


def evaluate(spec, lab, table, grid_rc, train_idx, test_idx, nx, ny, k=10, limit=None) -> dict:
    shots = pick_shots_spread(spec, lab, train_idx, table, k)
    te = test_idx if limit is None else test_idx[:limit]
    errs, rows = [], []
    for i in te:
        r = few_shot_predict(spec[i], shots, grid_rc, nx, ny)
        rows.append({"sample_id": table[i].sample_id, **r})
        if not r.get("available"):
            return {"available": False, "reason": r.get("reason"), "rows": rows}
        if lab["presence"][i] == 1:
            errs.append((abs(float(r["x"]) - table[i].x), abs(float(r["y"]) - table[i].y)))
    e = np.asarray(errs) if errs else np.zeros((0, 2))
    return {"available": True, "pos_mae": float(e.mean()) if len(e) else None,
            "within1_acc": float(np.all(e <= 1, axis=1).mean()) if len(e) else None,
            "exact_acc": float(np.all(np.round(e) == 0, axis=1).mean()) if len(e) else None,
            "n": len(te), "k_shots": k, "rows": rows}
