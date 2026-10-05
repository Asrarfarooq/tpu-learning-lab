#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="torchtpu-learning-lab"
NETWORK_NAME="tpu-lab-vpc"
SUBNET_NAME="tpu-lab-subnet"
REGION="${1:-us-east1}"

case "${REGION}" in
  us-central1)   SUBNET_CIDR="10.128.0.0/20" ;;
  us-central2)   SUBNET_CIDR="10.128.16.0/20" ;;
  us-east1)      SUBNET_CIDR="10.128.32.0/20" ;;
  us-east5)      SUBNET_CIDR="10.128.48.0/20" ;;
  europe-west4)  SUBNET_CIDR="10.128.64.0/20" ;;
  *)             SUBNET_CIDR="10.128.128.0/20" ;;
esac

echo "[1/4] Verifying project access (${PROJECT_ID})..."
gcloud projects describe "${PROJECT_ID}" --format="yaml(projectId,projectNumber,lifecycleState)"

echo "[2/4] Enabling Compute Engine, Cloud TPU, and Storage APIs on ${PROJECT_ID}..."
gcloud services enable \
  compute.googleapis.com \
  tpu.googleapis.com \
  serviceusage.googleapis.com \
  storage.googleapis.com \
  --project="${PROJECT_ID}"

echo "[3/4] Ensuring VPC (${NETWORK_NAME}), subnet (${SUBNET_NAME}-${REGION}: ${SUBNET_CIDR}), and SSH/ICI firewall exist..."
if ! gcloud compute networks describe "${NETWORK_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute networks create "${NETWORK_NAME}" \
    --project="${PROJECT_ID}" \
    --subnet-mode=custom \
    --mtu=8896
fi

if ! gcloud compute networks subnets describe "${SUBNET_NAME}-${REGION}" --region="${REGION}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute networks subnets create "${SUBNET_NAME}-${REGION}" \
    --project="${PROJECT_ID}" \
    --network="${NETWORK_NAME}" \
    --region="${REGION}" \
    --range="${SUBNET_CIDR}" \
    --enable-private-ip-google-access
fi

if ! gcloud compute firewall-rules describe "${NETWORK_NAME}-allow-ssh-internal" --project="${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud compute firewall-rules create "${NETWORK_NAME}-allow-ssh-internal" \
    --project="${PROJECT_ID}" \
    --network="${NETWORK_NAME}" \
    --allow="tcp:22,tcp:8470-8479,tcp:8431,icmp" \
    --source-ranges="0.0.0.0/0"
fi

echo "[4/4] Verifying v6e and v5p Spot quotas on ${PROJECT_ID}..."
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")"
gcloud alpha services quota list \
  --service=tpu.googleapis.com \
  --consumer="projects/${PROJECT_NUMBER}" \
  --filter="metric:(tpu-v6e-preemptible OR tpu-v5p-preemptible)" \
  --format="table(metric,consumerQuotaLimits[0].quotaBuckets[0].effectiveLimit:label=SPOT_CHIP_LIMIT)" || true

echo "Bootstrap complete for ${PROJECT_ID} in region ${REGION} (${SUBNET_CIDR})."
