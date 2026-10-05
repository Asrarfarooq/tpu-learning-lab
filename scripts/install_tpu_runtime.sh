#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="torchtpu-learning-lab"
ZONE="${1:-us-central1-b}"
NODE_ID="${2:-tpu-lab-vm}"
CORP_ACCOUNT="${3:-asrarfarooq@google.com}"

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE_DIR="${LAB_ROOT}/.wheelhouse"
mkdir -p "${WHEELHOUSE_DIR}"

echo "[1/3] Downloading torch_tpu and dependencies on Cloudtop into ${WHEELHOUSE_DIR}..."
echo "      (Using corporate account ${CORP_ACCOUNT} locally; token never leaves Cloudtop)"
AR_TOKEN="$(gcloud auth print-access-token --account="${CORP_ACCOUNT}")"
AR_INDEX_URL="https://oauth2accesstoken:${AR_TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"

python3 -m pip download \
  --dest "${WHEELHOUSE_DIR}" \
  --pre \
  --python-version 3.12 \
  --platform manylinux_2_28_x86_64 \
  --only-binary=:all: \
  --index-url "${AR_INDEX_URL}" \
  --extra-index-url "https://pypi.org/simple" \
  torch_tpu

unset AR_TOKEN
unset AR_INDEX_URL

echo "[2/3] Copying .wheelhouse/ to TPU VM ${NODE_ID} (${ZONE}, project=${PROJECT_ID})..."
gcloud compute tpus tpu-vm ssh "${NODE_ID}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command="mkdir -p ~/wheelhouse"

gcloud compute tpus tpu-vm scp --recurse \
  "${WHEELHOUSE_DIR}/"* \
  "${NODE_ID}:~/wheelhouse/" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}"

echo "[3/3] Installing Python 3.12, torch_tpu (from local wheelhouse), and jax[tpu] on ${NODE_ID}..."
gcloud compute tpus tpu-vm ssh "${NODE_ID}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --command='
    set -euo pipefail
    if ! command -v python3.12 >/dev/null 2>&1; then
      sudo apt-get update -y
      sudo apt-get install -y software-properties-common
      sudo add-apt-repository -y ppa:deadsnakes/ppa || true
      sudo apt-get update -y
      sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    fi
    if [ ! -d ~/tpu-env ]; then
      python3.12 -m venv ~/tpu-env
    fi
    source ~/tpu-env/bin/activate
    pip install --upgrade pip
    pip install --pre --find-links ~/wheelhouse torch_tpu
    pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install flax optax numpy tabulate tensorboard tbp-nightly
    python -c "import torch, jax; print(\"Torch version:\", torch.__version__, \"| JAX version:\", jax.__version__)"
  '

echo "TPU runtime installation complete on ${NODE_ID}."
