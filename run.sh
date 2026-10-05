#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLOUDSDK_CORE_PROJECT="torchtpu-learning-lab"
export TPU_LAB_PROJECT="torchtpu-learning-lab"
export TPU_LAB_HOST="127.0.0.1"
export TPU_LAB_PORT="${1:-8765}"

if [[ -x "/usr/local/google/home/asrarfarooq/.pyenv/versions/axlearn-env/bin/python3" ]]; then
  export TPU_LAB_LOCAL_PYTHON="/usr/local/google/home/asrarfarooq/.pyenv/versions/axlearn-env/bin/python3"
else
  export TPU_LAB_LOCAL_PYTHON="python3"
fi

echo "Starting TorchTPU & JAX Learning Lab on http://${TPU_LAB_HOST}:${TPU_LAB_PORT}"
echo "Session-isolated GCP Project: ${CLOUDSDK_CORE_PROJECT}"
echo "Local Python Runner: ${TPU_LAB_LOCAL_PYTHON}"
exec "${TPU_LAB_LOCAL_PYTHON}" "${LAB_ROOT}/backend/main.py"
