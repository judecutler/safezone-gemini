# Google tools used in SafeZone-Gemini

| Tool | Where | Why it is load-bearing |
|---|---|---|
| **Gemini API, `gemini-3.8-flash`** (AI Studio key) | `safezone/gemini.py` | Every AI decision in the product. Detection + segmentation masks of 13 object types in one `generateContent` call; room reasoning and the go-bag check as structured JSON (`responseJsonSchema`, `thinkingLevel: low`). Replaces two Meta models and a custom wire-grammar parser. |
| **Antigravity CLI (`agy`)** | git history of this repo | Every commit in the hackathon window was made through agy sessions (see `docs/agy-prompts/`). |
| **Cloud Run** | `Dockerfile`, `scripts/deploy.sh` | Hosted prototype, `--min-instances 1` so the judge's first scan is not a cold start. |
| **Cloud Build / Artifact Registry** | `gcloud run deploy --source .` | Builds the container from the repo. |
| **Secret Manager** | `scripts/deploy.sh` | `GEMINI_API_KEY` is never in the image or the repo. |

Not used on purpose: Gemini TTS. The emergency voice must work with no network while the ground is shaking, so it stays on the phone's own speech engine.
