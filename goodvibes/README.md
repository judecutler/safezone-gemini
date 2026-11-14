# GoodVibes — see inside sealed containers, without touching them

A laser grid and a camera read the microscopic surface vibrations of a sealed, opaque container.
A **Vibration Transformer** turns those vibrations into what is inside. **Gemini** inspects alongside it —
as a zero-training few-shot analyst, as an embedding-kNN baseline, and as the inspection agent that explains
every call with the laser points and frequency bands that drove it.

Built for the Google DeepMind × Berkeley Hackathon (Nov 14, 2026). Track: Antigravity CLI.

## Why

Conventional vision reads surfaces. Incumbent pharma inspection (laser headspace analysis) reads only the gas above
the product. Nobody reads the *product* inside a sealed container non-destructively at line speed. Surface vibration
carries that information (Kichler, Bagon, Sheinin — ICCV 2025); GoodVibes is the inference layer on top of it.

## What the prototype does

- Data: Weizmann `eturok-weizmann/laser-vibrations` — 10×10 laser grid on a sealed cardboard box, 4 speakers,
  50–1000 Hz chirp, a hidden object at one of ~130 positions on an 11×12 grid (plus one empty reference per speaker). 551 recordings.
- `ml/` — spectra on a fixed 1 Hz grid → Vibration Transformer (PointTransformer → ShapeTransformer) →
  continuous position, a localization confidence, and a 10×10 occupancy heatmap on the laser grid.
  Held out by *physics*: S1 unseen excitation (speaker), S2 unseen location, S3 random.
- `gemini/` — `gemini-3.8-flash` few-shot on rendered vibration signatures (structured output);
  `gemini-embedding-2` + kNN; function-calling inspection agent with a structured report; multimodal self-audit.
- `app/` — FastAPI + single-page UI: an autonomous line monitor — every container auto-inspected, PASS/FLAG with a reject queue, batch-record CSV + QA alert; click any container for heatmap, ground-truth reveal, spectra, and the Gemini report.

## Results (real data)

| model (S3: fixed line excitation, 110 held-out containers) | pos MAE (cells) | within 1 cell | exact cell |
|---|---|---|---|
| Vibration Transformer (ours, differential input) | 1.34 | 43% | 21% |
| Gradient boosting on band energy | 1.42 | 33% | 10% |
| kNN on band energy | 1.70 | 24% | 7% |
| Ridge on band energy | 1.73 | 25% | 10% |
| Random guess on an 11×12 grid | ≈3.8 | ≈7% | <1% |

Differential input = each recording minus the empty-container reference taken with the same excitation (one per speaker in the dataset). S1 (an excitation the model never heard) remains the hard open case and is reported as such in `results/metrics.json` once trained.

## Run

```bash
pip install -r requirements.txt
bash scripts/download.sh                                   # laptop; HF is blocked in some sandboxes
python -m ml.dataset --root data/train --out results/features.npz
python -m ml.train --split all --features results/features.npz   # writes results/best_S*.pt, metrics.json
GEMINI_API_KEY=... python -m gemini.evaluate --splits S1 --limit 40   # Gemini rows on the scoreboard
GEMINI_API_KEY=... uvicorn app.main:app --port 8080        # http://localhost:8080
```

No data yet? `python -m ml.synth && python -m ml.dataset --root data/synth --out results/features.npz` runs the
whole stack on physically-flavored synthetic recordings.

Deploy: `bash scripts/deploy.sh` (Cloud Run, us-central1, key from Secret Manager).

## Google tools used

Antigravity CLI (`agy`) · Gemini API: `gemini-3.8-flash` (agent, few-shot, self-audit), `gemini-embedding-2` (kNN) ·
Cloud Run · Cloud Build · Secret Manager · Google AI Studio (key, prompt iteration).

## Honest scope

The demo proves *hidden-contents detection and localization* inside a sealed opaque container. Fill level,
lyophilized-cake state, viscosity and aggregation in biologics are the same measurement with new labels — that is
the roadmap, not the demo. Industrial / pharmaceutical inspection only.
