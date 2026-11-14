#!/usr/bin/env bash
# Deploy SafeZone-Gemini to Cloud Run. Needs: gcloud authed, a project with billing, GEMINI_API_KEY in the env or .env.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT="${GCP_PROJECT:-gen-lang-client-0623180044}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE:-safezone-gemini}"
if [ -z "${GEMINI_API_KEY:-}" ] && [ -f .env ]; then
  GEMINI_API_KEY="$(grep -E '^GEMINI_API_KEY=' .env | cut -d= -f2- | tr -d "'\"\r" )"
fi
[ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY is not set"; exit 1; }
gcloud config set project "$PROJECT" >/dev/null
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com >/dev/null
if ! gcloud secrets describe GEMINI_API_KEY >/dev/null 2>&1; then
  printf '%s' "$GEMINI_API_KEY" | gcloud secrets create GEMINI_API_KEY --replication-policy=automatic --data-file=-
else
  printf '%s' "$GEMINI_API_KEY" | gcloud secrets versions add GEMINI_API_KEY --data-file=- >/dev/null
fi
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" >/dev/null
gcloud run deploy "$SERVICE" --source . --region "$REGION" --allow-unauthenticated \
  --memory 1Gi --cpu 1 --min-instances 1 --timeout 300 \
  --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest" \
  --set-env-vars "GEMINI_MODEL=${GEMINI_MODEL:-gemini-3.8-flash}"
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
