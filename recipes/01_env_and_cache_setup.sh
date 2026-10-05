#!/usr/bin/env bash
# Recipe 1: Verified TorchTPU Environment, 3-Tier Compilation Cache & Distributed Setup
# Grounded in //depot/google3/third_party/py/torch_tpu/csrc/common/env_vars.h

export TORCH_TPU_TIER2_COMPILATION_CACHE="torchtpu_shared_shm_cache"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="/tmp/torchtpu_persistent_cache"
mkdir -p "${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}"

# Optional: Enable XProf output directory when profiling
export TORCH_TPU_PROFILER_OUTPUT_DIR="/tmp/torchtpu_xprof_traces"
mkdir -p "${TORCH_TPU_PROFILER_OUTPUT_DIR}"

# Export distributed topology variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
if python3 -c "import torch, torch_tpu" >/dev/null 2>&1; then
  eval $(python3 -m torch.tpu.distributed | sed 's/^/export /')
fi

echo "TorchTPU environment configured:"
echo "  TORCH_TPU_TIER2_COMPILATION_CACHE      = ${TORCH_TPU_TIER2_COMPILATION_CACHE}"
echo "  TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT = ${TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT}"
