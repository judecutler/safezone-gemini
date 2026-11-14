"""GoodVibes API + static UI. Run: uvicorn app.main:app --port 8080"""
from __future__ import annotations

import glob
import io
import json
import os
import random

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ml.infer import Inferencer
from gemini.agent import Agent
from gemini.render import signature_png, heatmap_png
from gemini.client import client as gemini_client
from gemini.notify import batch_log, notify_qa

CKPT = os.environ.get("GV_CKPT", "results/best_S1.pt")
FEATS = os.environ.get("GV_FEATURES", "results/features.npz")
STATIC = os.path.join(os.path.dirname(__file__), "frontend")

app = FastAPI(title="GoodVibes", version="0.1")
INF: Inferencer | None = None
AGENT: Agent | None = None
_cache: dict = {}


@app.on_event("startup")
def _load():
    global INF, AGENT
    if not os.path.exists(CKPT) or not os.path.exists(FEATS):
        raise RuntimeError(f"missing {CKPT} or {FEATS}; run ml.dataset + ml.train first")
    INF = Inferencer(CKPT, FEATS)
    AGENT = Agent(INF)
    INF.predict(INF.table[0].sample_id)   # warm


@app.get("/healthz")
def healthz():
    return {"ok": True, "gemini": gemini_client() is not None, "n_samples": len(INF.table)}


@app.get("/api/samples")
def samples(split: str = "S1", part: str = "test"):
    return INF.samples(split, part)


@app.get("/api/random")
def random_sample(split: str = "S1"):
    return random.choice(INF.samples(split, "test"))


@app.get("/api/stream")
def stream(split: str = "S1", seed: int | None = None):
    """Server-shuffled sequence of held-out containers for the autonomous line monitor."""
    items = INF.samples(split, "test")
    key = ("conf", split)
    if key not in _cache:
        confs = np.array([INF.predict(it["sample_id"])["pred"]["confidence"] for it in items])
        _cache[key] = float(np.percentile(confs, 15)) if len(confs) else 0.0
    rng = random.Random(seed)
    rng.shuffle(items)
    return {"split": split, "items": items, "nx": INF.nx, "ny": INF.ny, "flag_threshold": _cache[key]}


class InferReq(BaseModel):
    sample_id: int
    with_gemini: bool = True


@app.post("/api/infer")
def infer(req: InferReq):
    if req.sample_id not in INF.by_id:
        raise HTTPException(404, "unknown sample")
    out = INF.predict(req.sample_id)
    out["compare"] = INF.compare_with_empty(req.sample_id)
    cached = _cache.get(("fewshot", req.sample_id))
    out["gemini_fewshot"] = cached if cached else {"available": False, "reason": "not cached; run gemini.listener evaluate"}
    return out


@app.get("/api/truth/{sample_id}")
def truth(sample_id: int):
    t = INF.truth(sample_id)
    t["has_overhead"] = bool(t.get("overhead_path"))
    t.pop("overhead_path", None); t.pop("mask_path", None)
    return t


@app.get("/api/overhead/{sample_id}")
def overhead(sample_id: int):
    p = INF.overhead_path(sample_id)
    if not p or not os.path.exists(p):
        raise HTTPException(404, "no overhead image")
    return FileResponse(p)


@app.get("/api/spectrum/{sample_id}/{laser_idx}")
def spectrum(sample_id: int, laser_idx: int):
    return INF.point_spectrum(sample_id, laser_idx)


@app.get("/api/signature/{sample_id}.png")
def signature(sample_id: int):
    i = INF.by_id[sample_id]
    return Response(signature_png(INF.spec[i], INF.grid_rc), media_type="image/png")


@app.get("/api/heatmap/{sample_id}.png")
def heatmap(sample_id: int):
    p = INF.predict(sample_id)["pred"]
    return Response(heatmap_png(np.asarray(p["occ"]), INF.grid_rc), media_type="image/png")


@app.get("/api/metrics")
def metrics():
    m = INF.metrics()
    extra = "results/metrics_gemini.json"
    if os.path.exists(extra):
        m["gemini"] = json.load(open(extra))
    return m


class DispositionReq(BaseModel):
    sample_id: int
    disposition: str          # "PASS" | "FLAG"


@app.post("/api/disposition")
def disposition(req: DispositionReq):
    """Autonomous line decision → batch record (+ QA channel for flags). Called by the UI for every container."""
    pred = INF.predict(req.sample_id)["pred"]
    out = {"batch_log": batch_log(req.sample_id, pred, req.disposition)}
    if req.disposition == "FLAG":
        out["qa_notify"] = notify_qa(req.sample_id, pred)
    return out


@app.post("/api/report")
def report(req: InferReq):
    r = AGENT.report(req.sample_id)
    if r.get("available"):
        batch_log(req.sample_id, INF.predict(req.sample_id)["pred"], "REPORT", r.get("report"))
    return r


@app.post("/api/audit")
def audit(req: InferReq):
    return AGENT.self_audit(req.sample_id)


class ChatReq(BaseModel):
    sample_id: int
    message: str
    history: list[dict] = []
    allow_truth: bool = False


@app.post("/api/chat")
def chat(req: ChatReq):
    return AGENT.chat(req.sample_id, req.history, req.message, req.allow_truth)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), fs_hz: float = Form(...)):
    """Bring your own recording: .npz/.npy with shifts shaped [n_lasers, T, 2] (any axis order)."""
    raw = await file.read()
    buf = io.BytesIO(raw)
    if file.filename.endswith(".npz"):
        z = np.load(buf)
        arrs = [z[k] for k in z.files if z[k].ndim == 3]
        if not arrs:
            raise HTTPException(400, "no 3-D array in npz")
        arr = arrs[0]
    else:
        arr = np.load(buf)
    try:
        return INF.predict_upload(arr, fs_hz)
    except ValueError as e:
        raise HTTPException(400, str(e))


if os.path.isdir(STATIC):
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
