# Unit 5.2: TorchTPU `DDP` / `FSDP` vs. JAX `NamedSharding` & End-to-End Transformer Benchmark

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Write multi-chip distributed training scripts in **TorchTPU** (`python -m torch.tpu.distributed`, `dist.init_process_group(backend="tpu_dist")`, `DDP`, and `FSDP`) and **JAX** (`Mesh`, `NamedSharding`, `PartitionSpec`).
2. Run a complete end-to-end **Transformer Decoder (RMSNorm + Multi-Head SDPA Attention + SwiGLU FFN)** benchmark comparing TorchTPU and JAX throughput (tokens/sec and TFLOPs/sec).

## 2. Side-by-Side Multi-Chip Code Patterns

### TorchTPU (`torchrun --nnodes=1 --nproc_per_node=4`)
Per `docs/markdown/distributed_training_101.md` (`go/tt-devguide`), before launching `torchrun`, export the multi-process PJRT port and process count environment variables via `python -m torch.tpu.distributed`:
```bash
eval $(python -m torch.tpu.distributed | sed 's/^/export /')
torchrun --nnodes=1 --nproc_per_node=4 train_ddp.py
```
Inside `train_ddp.py`, initialize the native `"tpu_dist"` collective backend. Note that in multi-process mode, each worker process is bound exclusively to its assigned chip as `torch.device("tpu")` (index 0 within that process):
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize native TorchTPU PJRT collective backend over ICI
dist.init_process_group(backend="tpu_dist")

# Each worker process sees its own assigned TPU chip as 'tpu' (device index 0)
device = torch.device("tpu")
model = MyTransformer().to(device)
ddp_model = DDP(model, gradient_as_bucket_view=True)
compiled_model = torch.compile(ddp_model, backend="tpu")
```

### JAX Single-Controller GSPMD (`NamedSharding`)
Unlike PyTorch's multi-process `torchrun` (which spawns 1 Python process per chip on a single host), JAX uses a **single Python process per host** that controls all attached TPU chips simultaneously via **GSPMD** (General Single-Program Multiple-Data):
```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = jax.devices()
mesh = Mesh(devices, ("data",))

# Shard batch dimension across 'data' axis; replicate hidden dimensions
batch_sharding = NamedSharding(mesh, P("data", None))
replicated_sharding = NamedSharding(mesh, P())
x_sharded = jax.device_put(x_batch, batch_sharding)
params_replicated = jax.device_put(params, replicated_sharding)
```

## 3. Guided Hands-On Lab (`labs/lab_02_transformer_inference_benchmark.py`)

Click **Run Code** in the right-hand pane to run the capstone **End-to-End Transformer Decoder Benchmark (`labs/lab_02_transformer_inference_benchmark.py`)**, which builds a multi-layer Decoder-only Transformer (Multi-Head Causal Attention + RMSNorm + SwiGLU) in both **PyTorch (`torch_tpu`)** and **JAX**, measures tokens/second and TFLOPs/second, and verifies numerical parity.
