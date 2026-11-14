# GoodVibes — agent context (read fully before any work)

Non-contact inspection of sealed opaque containers from laser-speckle surface vibrations.
Google DeepMind × Berkeley Hackathon, Antigravity CLI track. Submission Sat Nov 14, 10:30 AM PT.
Scoring: Gemini integration 40%, wow 30%, impact 20%, communication 10%. Gemini must be load-bearing.

## Data (Hugging Face, Weizmann)
`eturok-weizmann/laser-vibrations` (551 samples). 10×10 laser grid on the side wall of a sealed cardboard box;
4 loudspeakers (4-bit code `speakers`, e.g. `0001`), 50–1000 Hz log chirp, 3.2 s. Hidden object = `cube` or `empty`,
at `x_position` (column) / `y_position` (row) on a coarse position grid; overhead `mask.png` + `x_com,y_com`.

Verified from `manifest.json`:
- `processing_config.speckle_shifts.fs_hz = 2500`; shifts shape `[100 lasers, 9000 frames, 2 (x,y)]`
- `speckle_shifts_clean`: Butterworth bandpass 50–1000 Hz order 5, then Hann (`apply_order: filter_then_hann`)
- `speckle_shifts_fft`: rfft along axis 1, cropped to 50–1000 Hz, complex64
- laser grid geometry: `experiment_output.laser_grid.selected_column_centers_x` (10 cols) and
  `selected_row_points_image_xy` (10 rows); laser index order must be derived from these, not assumed.
- tree: `data/metadata.jsonl`, `data/<7-digit id>/{manifest.json, speckle_shifts*.npz, ...}`,
  `data/image/<object>-<x>x-<y>y-<n>obj-<material>-<date>/{raw_overhead,cropped_overhead,segmented_overhead,mask}.png`
- the 5-sample `laser-vibrations-test` repo has NO npz files (only one laser as audio + preview mp4) — use the main repo.

**Real label distribution (verified on all 551):** 547 `cube`, only 4 `empty` → presence detection is degenerate and
must NOT be a headline metric. Positions form an 11×12 grid (`x_position` 0–10, `y_position` 0–11), ~130 distinct
cells, each recorded once per speaker code (4 recordings per cell; speakers 137/138/137/139). **The real task is
fine-grained localization:** predict continuous (x, y). S1 = same positions, never-heard excitation; S2 = never-seen
positions (interpolation). Headline metrics: `pos_mae` (L1 in grid units, mean over x,y), `within1_acc`
(|dx|≤1 and |dy|≤1), `exact_acc` (nearest cell == truth). A 130-way classifier with 3 examples per class is the
wrong tool; keep `cell` CE only as an auxiliary loss if at all.
Images live at repo-root `image/<dir>/` (not `data/image/`), per-sample `data/<id>/overhead.png`; metadata file
fields are full HF URLs — take the path after `/resolve/main/`.

Local layout: `data/train/` (main repo, small files only), `data/test/`. Both gitignored.

## Target system
1. `ml/features.py` — frozen spec: per laser, per axis, `|rfft|` on a fixed 1 Hz grid 50–999 Hz (950 bins) with a
   Hann window, `log1p(1e3·mag)`; tensor per sample `[100, 2, 950]`; band-energy map `[100, 5]` over bands
   50–100, 100–200, 200–400, 400–700, 700–1000 Hz. Cache all samples to `results/features.npz` (fp16 spectra).
2. `ml/dataset.py` — parse metadata/manifests → table; labels: `presence`, `cell` (index over observed (x,y)),
   `com`, `occ` (cell footprint on the 10×10 laser grid). Splits held out by physics:
   S1 unseen excitation (train speakers {0001,0010,0100}, test {1000}); S2 unseen location (20% of cells); S3 random.
3. `ml/baselines.py` — logistic regression on band-energy maps (StandardScaler). The floor to beat.
4. `ml/model.py` — Vibration Transformer (Kichler et al. ICCV'25): 2×50 spectral patches → 19 tokens/point →
   shared PointTransformer with a [pnt] token → 100 [pnt] tokens + learned row/col embeddings + [cls] →
   ShapeTransformer → heads: **pos(2, sigmoid) = (x/(nx-1), y/(ny-1)) — primary**, occ(1 per point; target = Gaussian bump σ≈1 cell
   around the true position rendered on the 10×10 laser grid), presence(2) and cell(K) as auxiliaries.
   Loss = 1.0·SmoothL1(pos) + 0.5·BCE(occ) + 0.2·CE(cell) + 0.1·CE(presence); mask position terms for empties.
   **Critical, learned the hard way:** standardize features per-feature across the training set (store mu/sd in
   the checkpoint, apply at inference). Per-sample z-scoring alone hides the per-point spatial contrast and the
   model never learns position. Augment: smooth random spectral filter (±3 dB), 50% token dropout, 10% point
   dropout, ±2-bin shift, noise. AdamW 3e-4..5e-4, wd 0.05, OneCycle, bs 16, d=256 depth 4 on GPU
   (d=128 depth 2 on CPU), early stop on val loss. Save `results/best_<split>.pt`, `results/metrics.json`
   (baseline vs transformer × S1/S2/S3: pos_mae, within1_acc, exact_acc, occ_iou), `results/preds/*.json`.
5. `gemini/` (all via `google-genai`, key from env `GEMINI_API_KEY`, never in code; every call must degrade to
   `{"available": false, "reason"}` so the demo never 500s):
   - `listener.py`: `gemini-3.8-flash` few-shot on a rendered PNG signature (5 band maps + per-row spectra) with
     `response_schema {presence, x, y, confidence, reasoning}`; score it on S1 test like any model.
   - `embed_knn.py`: `gemini-embedding-2` on the same PNG → cosine kNN (k=5); score it.
   - `agent.py`: function-calling inspector (tools: run_inference, compare_with_empty_reference,
     get_point_spectrum, get_metrics, get_ground_truth only when revealed) → structured `InspectionReport`
     {verdict, location_cell, confidence, evidence[{laser_idx, band_hz, delta_db}], recommended_action, caveats};
     plus free-form chat. `self_audit`: predicted heatmap PNG vs overhead photo → {agree, note}.
   - `evaluate.py` → `results/metrics_gemini.json`.
6. `app/main.py` FastAPI: `/api/samples`, `/api/stream?split=S1` (server-shuffled sequence of held-out ids),
   `/api/infer`, `/api/truth/{id}`, `/api/overhead/{id}`, `/api/spectrum/{id}/{laser}`, `/api/signature/{id}.png`,
   `/api/metrics`, `/api/report`, `/api/audit`, `/api/chat`, `/api/upload` (bring-your-own shifts npz + fs_hz),
   `/healthz`. Single-file `app/frontend/index.html` (no build step), dark industrial console. **It is an autonomous
   line monitor, not a click-to-demo:** on load, containers (held-out recordings) move along a conveyor, one every
   ~4 s, each auto-inspected as it passes — 10×10 heatmap, verdict, confidence; PASS rolls on, FLAG drops into a
   reject queue on the right with the Gemini report attached. Counters: inspected / flagged / per-minute throughput.
   Clicking any container pauses the line and opens the inspector: predicted vs ground truth (Reveal), click-a-laser
   spectrum vs empty reference, Gemini report + chat, signature PNG. Scoreboard panel below. Top strip shows three
   tiers: Proved tonight / Same measurement, next labels / Frontier.
7. `Dockerfile` (python:3.11-slim, CPU torch, weights + features + metrics baked in), `scripts/deploy.sh`
   (Cloud Run us-central1, `--min-instances 1`, key via Secret Manager), `cloudbuild.yaml`.

## Conventions
- Python 3.11+, type hints, small modules, no notebooks. Verify every step on the real artifact (run it).
- Commit after each working unit with a descriptive message. Never commit data, keys, or `.pt` files.
- Never use the words "listen in", "eavesdrop", or "speech". Industrial / pharmaceutical inspection only.
- Model IDs: `gemini-3.8-flash`, `gemini-embedding-2`. Verify the multimodal `embed_content` call shape in docs.
