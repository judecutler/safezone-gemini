#!/usr/bin/env bash
# One-shot deploy from a laptop. Needs: gcloud auth login; a project with billing; results/best_S1.pt + features.npz.
set -euo pipefail
PROJECT=${PROJECT:-$(gcloud config get-value project)}
REGION=us-central1
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com --project "$PROJECT"
if ! gcloud secrets describe GEMINI_API_KEY --project "$PROJECT" >/dev/null 2>&1; then
  : "${GEMINI_API_KEY:?set GEMINI_API_KEY in the environment for the first deploy}"
  printf '%s' "$GEMINI_API_KEY" | gcloud secrets create GEMINI_API_KEY --data-file=- --project "$PROJECT"
fi
PN=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding GEMINI_API_KEY --member="serviceAccount:${PN}-compute@developer.gserviceaccount.com" --role=roles/secretmanager.secretAccessor --project "$PROJECT" >/dev/null
gcloud run deploy goodvibes --source . --region "$REGION" --project "$PROJECT" --allow-unauthenticated \
  --memory 4Gi --cpu 2 --min-instances 1 --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest
gcloud run services describe goodvibes --region "$REGION" --project "$PROJECT" --format='value(status.url)'
