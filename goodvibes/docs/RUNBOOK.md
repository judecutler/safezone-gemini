# Morning runbook (Sat Nov 14) — from wake-up to submission

All commands in WSL from `/mnt/c/Users/roaan/goodvibes` with `source .venv/bin/activate`.

## 0. Unblock Gemini (2 min)
The project hit its monthly spend cap at ~04:00 (agy burned the $5 prepay). Go to https://ai.studio/spend →
project **GoodVibes** → raise the cap / add $10–20 credits. Until then every Gemini call in the product
degrades to "unavailable" (by design) and agy cannot run.

## 1. Sync the repo (1 min)
```bash
git pull --rebase origin main
pip install -r requirements.txt
```

## 1b. Train all splits on the laptop (10–20 min on a multi-core CPU; run in a second terminal while you do steps 2–3)
```bash
python -m ml.train --split all --features results/features.npz --epochs 150 --d 128 --depth 2 --lr 5e-4
```
This refreshes results/best_S1.pt, best_S2.pt, best_S3.pt and results/metrics.json (S1 unseen excitation, S2 unseen
location, S3 fixed line). The S3 checkpoint shipped overnight already works; this only improves the table.

## 2. Smoke test locally (3 min)
```bash
export GEMINI_API_KEY='...'
GV_CKPT=results/best_S3.pt uvicorn app.main:app --port 8080
```
Open http://localhost:8080 — the line should start moving on its own; click a container, Reveal, ask the inspector.

## 3. Gemini scoreboard rows (5 min, costs cents)
```bash
python -m gemini.evaluate --splits S3 --limit 30 --which fewshot
```

## 4. Deploy to Cloud Run (5–8 min)
```bash
bash scripts/deploy.sh      # prints the public URL; first run creates the GEMINI_API_KEY secret
```
Open the URL in an incognito window; let the line run; confirm the Gemini report appears.

## 5. Record (60 min)
Video 2 (1:00): screen-record the hosted URL — line running → click a flagged container → Reveal → spectrum → report.
Video 1 (2:00): you + script in docs/pitch.md. Upload both to YouTube (public). Share links.

## 6. Submit by 10:15
Form fields: team, one-pagers (docs/one-pager-*.md, paste as text), Cloud Run URL, both YouTube links,
GitHub URL (https://github.com/judecutler/goodvibes), track = AGY CLI, in-person = yes.

## If something breaks
- App won't start: check `results/best_S3.pt` and `results/features.npz` exist (features: `python -m ml.dataset --root data/train --out results/features.npz`).
- Gemini unavailable: spend cap (step 0) or key not exported.
- Cloud Run build fails on torch: the Dockerfile uses the CPU wheel index; retry once, Cloud Build is flaky on first pull.
