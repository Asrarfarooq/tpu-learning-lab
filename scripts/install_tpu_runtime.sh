#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="torchtpu-learning-lab"
ZONE="${1:-us-central1-b}"
NODE_ID="${2:-tpu-lab-vm}"
MODE="${3:---pypi}"

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE_DIR="${LAB_ROOT}/.wheelhouse"

if [[ "${MODE}" == "--from-wheelhouse" ]]; then
  echo "[1/2] Copying local custom wheels from ${WHEELHOUSE_DIR}/ to ${NODE_ID} (${ZONE}, project=${PROJECT_ID})..."
  gcloud compute tpus tpu-vm ssh "${NODE_ID}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --command="mkdir -p ~/wheelhouse"

  gcloud compute tpus tpu-vm scp --recurse \
    "${WHEELHOUSE_DIR}/"*.whl \
    "${NODE_ID}:~/wheelhouse/" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}"

  echo "[2/2] Installing Python 3.12, custom torch_tpu wheel from ~/wheelhouse, and jax[tpu] on ${NODE_ID}..."
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
      pip install --find-links ~/wheelhouse torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu
      pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      pip install flax optax numpy tabulate tensorboard tbp-nightly
      python -c "import torch, jax; print(\"Torch:\", torch.__version__, \"| Accelerator:\", torch.accelerator.current_accelerator(), \"| Devices:\", torch.accelerator.device_count(), \"| JAX:\", jax.__version__)"
    '
else
  echo "[1/1] Installing Python 3.12, public PyPI torch_tpu==0.2.0, and jax[tpu] directly on ${NODE_ID} (${ZONE}, project=${PROJECT_ID})..."
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
      pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
      pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
      pip install flax optax numpy tabulate tensorboard tbp-nightly
      python -c "import torch, torch_tpu, jax; props = torch.tpu.get_device_properties(); print(\"Torch:\", torch.__version__, \"| torch_tpu:\", torch_tpu.__version__, \"| Chip:\", props.name, \"| Devices:\", torch.accelerator.device_count(), \"| JAX:\", jax.__version__)"
    '
fi

echo "TPU runtime installation complete on ${NODE_ID}."
