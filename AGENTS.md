# SafeZone-Gemini: context for every Antigravity session

Read this before touching code. Work in small verifiable steps; run `python -m pytest -q` and `node --test web/tests/*.test.mjs` before every commit.

## What this is
A household earthquake + tsunami plan. Photograph each room, the way out, the go bag; Gemini finds the hazards and the spot to drop, cover and hold on; code decides what reaches the plan. Tsunami routing and shelters come from official data. In an emergency the phone talks you out, offline.

Prior work: https://github.com/judecutler/safezone (Meta SAM 3.1 + Muse Spark). This repo is the 24-hour hackathon rebuild of the AI layer on Google Gemini plus a Cloud Run deploy.

## Architecture (one rule: the model proposes, code decides)
- `safezone/transport.py`: every outside call. Live, recording, replay. Fixture key = hash of the request minus auth headers.
- `safezone/gemini.py`: the only module that knows the Gemini API. `GeminiClient.detect()` (boxes + masks, one call), `.vision_json()` (JSON under a schema). REST `generateContent` with `x-goog-api-key`, `responseMimeType=application/json`, `responseJsonSchema`, `thinkingConfig.thinkingLevel=low`.
- `safezone/room/segment.py`: detections → `Segment`s (dedupe by IoU, masks → SVG outlines with PIL).
- `safezone/room/classify.py`: reasoning call; drops unknown ids, enforces role/kind rules.
- `safezone/room/score.py`: rejects a proposed floor spot in a doorway / tip zone / under stored items / near glass; fixes from `rules.FIX_TABLE` only.
- `safezone/gobag.py`: Ready.gov 14 ids; never says a kit is complete.
- `safezone/app.py`: Starlette JSON API + static PWA in `web/` (vanilla JS, no build).
- `tools/samples/*`: hand-authored SAMPLE answers in Gemini wire format, so tests and demo run offline.
- `tools/record_fixtures.py`: records real Gemini answers for the demo photo into `fixtures/demo` and writes `demo/plan.json`.

## Env
`GEMINI_API_KEY` (never commit; `.env` is gitignored), `GEMINI_MODEL` (default `gemini-3.8-flash`), `SAFEZONE_FIXTURES=fixtures/demo` for offline replay.

## Deploy
`bash scripts/deploy.sh` → Cloud Run `safezone-gemini`, key from Secret Manager, `--min-instances 1`.

## Conventions
- Pydantic shapes in `safezone/shapes.py` are the contract; parse at the boundary, trust inside.
- Every value carries provenance (`live` / `recorded` / `sample`); the UI labels sample data.
- No prose that promises safety: "best spot we could find", never "the safest spot" (`tests/test_adversarial_safety.py`).
- Commit messages: imperative, one line of why.
