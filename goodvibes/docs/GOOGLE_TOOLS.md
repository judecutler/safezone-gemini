# Google tools used (proof for judges)

| Tool | Where | Evidence |
|---|---|---|
| Antigravity CLI (`agy`) | all code generation / edits | commit history, `AGENTS.md`, session screenshots in `docs/proof/` |
| Gemini `gemini-3.8-flash` | `gemini/listener.py` (few-shot inspector, structured output), `gemini/agent.py` (function-calling agent + report), self-audit | `/api/report`, `/api/chat`, `/api/audit`; scoreboard row "Gemini few-shot" |
| Gemini `gemini-embedding-2` | `gemini/embed_knn.py` | scoreboard row "Gemini Embedding 2 + kNN" |
| Cloud Run | hosting | public URL in submission |
| Cloud Build + Artifact Registry | `cloudbuild.yaml` | build logs |
| Secret Manager | `GEMINI_API_KEY` | `scripts/deploy.sh` |
| Google AI Studio | key + prompt iteration | — |

Add screenshots of `agy` sessions to `docs/proof/` before submitting.
