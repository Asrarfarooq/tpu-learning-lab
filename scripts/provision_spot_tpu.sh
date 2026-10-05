#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="torchtpu-learning-lab"
ACTION="${1:-status}"
ZONE="${2:-us-east1-d}"
ACCELERATOR_TYPE="${3:-v6e-1}"
NODE_ID="${4:-tpu-lab-vm}"
QR_ID="${NODE_ID}-qr"
NETWORK_NAME="tpu-lab-vpc"
REGION="${ZONE%-*}"
SUBNET_NAME="tpu-lab-subnet-${REGION}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${ACCELERATOR_TYPE}" == v6e* ]]; then
  RUNTIME_VERSION="v2-alpha-tpuv6e"
  FALLBACK_ZONES=("${ZONE}" "us-east1-d" "us-east5-a" "us-east5-b" "us-central1-b" "us-central2-b" "europe-west4-a")
elif [[ "${ACCELERATOR_TYPE}" == v5p* ]]; then
  RUNTIME_VERSION="v2-alpha-tpuv5"
  FALLBACK_ZONES=("${ZONE}" "us-east1-d" "us-east5-a" "us-east5-b" "europe-west4-b")
else
  echo "Error: Only v6e-* and v5p-* accelerator topologies are supported (got: ${ACCELERATOR_TYPE})." >&2
  exit 1
fi

case "${ACTION}" in
  create)
    echo "Ensuring regional subnet ${SUBNET_NAME} exists in ${REGION}..."
    bash "${SCRIPT_DIR}/bootstrap_project.sh" "${REGION}" >/dev/null
    echo "Submitting Spot Queued Resource ${QR_ID} (${ACCELERATOR_TYPE}, runtime=${RUNTIME_VERSION}) in ${ZONE}..."
    gcloud compute tpus queued-resources create "${QR_ID}" \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --node-id="${NODE_ID}" \
      --accelerator-type="${ACCELERATOR_TYPE}" \
      --runtime-version="${RUNTIME_VERSION}" \
      --network="${NETWORK_NAME}" \
      --subnetwork="${SUBNET_NAME}" \
      --spot
    ;;
  hunt)
    echo "Starting multi-zone Spot hunter for ${ACCELERATOR_TYPE} across verified zones..."
    SEEN=""
    for Z in "${FALLBACK_ZONES[@]}"; do
      if [[ " ${SEEN} " == *" ${Z} "* ]]; then
        continue
      fi
      SEEN="${SEEN} ${Z}"
      R="${Z%-*}"
      S="tpu-lab-subnet-${R}"
      echo "--- Checking zone ${Z} (subnet ${S}) ---"
      bash "${SCRIPT_DIR}/bootstrap_project.sh" "${R}" >/dev/null
      gcloud compute tpus queued-resources delete "${QR_ID}" --project="${PROJECT_ID}" --zone="${Z}" --force --quiet >/dev/null 2>&1 || true
      if gcloud compute tpus queued-resources create "${QR_ID}" \
        --project="${PROJECT_ID}" \
        --zone="${Z}" \
        --node-id="${NODE_ID}" \
        --accelerator-type="${ACCELERATOR_TYPE}" \
        --runtime-version="${RUNTIME_VERSION}" \
        --network="${NETWORK_NAME}" \
        --subnetwork="${S}" \
        --spot; then
        echo "Waiting up to 90s for ${QR_ID} in ${Z} to transition to ACTIVE..."
        for _ in {1..9}; do
          sleep 10
          STATE="$(gcloud compute tpus queued-resources describe "${QR_ID}" --project="${PROJECT_ID}" --zone="${Z}" --format="value(state.state)" 2>/dev/null || echo "UNKNOWN")"
          echo "  [${Z}] state=${STATE}"
          if [[ "${STATE}" == "ACTIVE" ]]; then
            echo "SUCCESS: Spot TPU ${NODE_ID} (${ACCELERATOR_TYPE}) is ACTIVE in zone ${Z}!"
            exit 0
          fi
          if [[ "${STATE}" == "FAILED" || "${STATE}" == "SUSPENDED" ]]; then
            break
          fi
        done
        echo "Zone ${Z} is queued (${STATE}); cleaning up ${QR_ID} in ${Z} and trying next zone..."
        gcloud compute tpus queued-resources delete "${QR_ID}" --project="${PROJECT_ID}" --zone="${Z}" --force --async --quiet >/dev/null 2>&1 || true
      fi
    done
    echo "No zone fulfilled ${ACCELERATOR_TYPE} immediately; leave a Queued Resource in your preferred zone via 'create' to wait in line."
    exit 1
    ;;
  status)
    echo "=== Queued Resources in ${ZONE} (${PROJECT_ID}) ==="
    gcloud compute tpus queued-resources list \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --format="table(name.basename(),state.state,tpu.nodeSpec[0].node.acceleratorType,tpu.nodeSpec[0].nodeId)" || true
    echo ""
    echo "=== Active TPU VMs in ${ZONE} (${PROJECT_ID}) ==="
    gcloud compute tpus tpu-vm list \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --format="table(name.basename(),acceleratorType,state,networkEndpoints[0].ipAddress,symptoms)" || true
    ;;
  delete)
    echo "Deleting Queued Resource ${QR_ID} and node ${NODE_ID} in ${ZONE} (${PROJECT_ID})..."
    gcloud compute tpus queued-resources delete "${QR_ID}" \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --force \
      --async \
      --quiet || gcloud compute tpus tpu-vm delete "${NODE_ID}" \
      --project="${PROJECT_ID}" \
      --zone="${ZONE}" \
      --quiet
    ;;
  *)
    echo "Usage: $0 {create|hunt|status|delete} [zone] [accelerator_type] [node_id]"
    exit 1
    ;;
esac
