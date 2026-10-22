#!/usr/bin/env bash
# Deploys the TorchTPU & JAX Learning Lab Dashboard to Cloud Run in Argolis project torchtpu-learning-lab.
# Enforces Argolis IAM policy: --no-allow-unauthenticated (never grants allUsers or @google.com).
set -euo pipefail

PROJECT_ID="torchtpu-learning-lab"
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
REGION="${1:-us-central1}"
SERVICE_NAME="torchtpu-learning-lab"
ARGOLIS_USER="asrarfarooq@gcp.altostrat.com"

export CLOUDSDK_CORE_PROJECT="${PROJECT_ID}"

echo "[1/4] Enabling Cloud Run, Cloud Build, and Artifact Registry APIs on ${PROJECT_ID}..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  iap.googleapis.com \
  --project="${PROJECT_ID}"

echo "[2/4] Granting Cloud Build source-deploy roles to ${COMPUTE_SA}..."
for ROLE in roles/cloudbuild.builds.builder roles/storage.objectAdmin roles/artifactregistry.writer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="${ROLE}" \
    --quiet >/dev/null
done

echo "[3/4] Deploying ${SERVICE_NAME} to Cloud Run (${REGION}) with --no-allow-unauthenticated..."
gcloud run deploy "${SERVICE_NAME}" \
  --source=. \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --no-allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --set-env-vars="TPU_LAB_PROJECT=${PROJECT_ID},CLOUDSDK_CORE_PROJECT=${PROJECT_ID}" \
  --quiet

echo "[4/4] Granting roles/run.invoker exclusively to ${ARGOLIS_USER}..."
gcloud run services add-iam-policy-binding "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --member="user:${ARGOLIS_USER}" \
  --role="roles/run.invoker"

echo "Deployment complete. To access locally with your Argolis identity before Cloud IAP is enabled:"
echo "  gcloud run services proxy ${SERVICE_NAME} --project=${PROJECT_ID} --region=${REGION} --port=8766"
