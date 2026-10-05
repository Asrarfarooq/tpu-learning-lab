# Unit 3.2: Capturing & Reading XProf Hardware Traces

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Capture hardware-level **XProf** profiles in both **TorchTPU** (`torch.profiler.profile` with `TpuProfilerConfig` and `TORCH_TPU_PROFILER_OUTPUT_DIR`) and **JAX** (`jax.profiler.trace` + `StepTraceAnnotation`).
2. Interpret the 4 essential XProf tools in TensorBoard / Profile Viewer:
   * **Overview Page** (Step Time Breakdown, FLOPs Utilization, Input-Bound vs. Compute-Bound %)
   * **Trace Viewer** (Host CPU threads vs. TPU TensorCore timeline, DMA transfers, HLO Op rows)
   * **HLO Op Profile** (Exact FLOPs, HBM bandwidth utilization, and time % per HLO category: `dot`, `fusion`, `all-reduce`)
   * **Memory Viewer** (Peak HBM allocation timeline and buffer fragmentation)

## 2. Capturing XProf Traces in TorchTPU (`go/tt-devguide`)

### 1. Environment Variables for Output Directory & Session ID
TorchTPU exposes two environment variables in `csrc/common/env_vars.h` to control where `.xplane.pb` traces are written when no explicit trace handler callback is passed:
```bash
export TORCH_TPU_PROFILER_OUTPUT_DIR="/tmp/torchtpu_xprof_logs"
export TORCH_TPU_XPROF_SESSION_ID="benchmark_run_01"
```

### 2. Programmatic `torch.profiler` + `TpuProfilerConfig` Context Manager
TorchTPU integrates directly with `torch.profiler` (`ProfilerActivity.TPU`) and provides `torch.tpu.profiler.TpuProfilerConfig` for configuring host and device tracer verbosity:
```python
import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from torch.tpu.profiler import TpuProfilerConfig

prof_schedule = schedule(wait=2, warmup=2, active=3, repeat=1)
tpu_config = TpuProfilerConfig(host_tracer_level=2, device_tracer_level=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.TPU],
    schedule=prof_schedule,
    experimental_config=tpu_config,
    on_trace_ready=tensorboard_trace_handler("/tmp/torchtpu_xprof_logs"),
) as prof:
    for step in range(10):
        train_step()
        torch.accelerator.synchronize()
        prof.step()
```

### 3. Capturing XProf Traces in JAX
```python
import jax

with jax.profiler.trace("/tmp/jax_xprof_logs"):
    for step in range(5):
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            params, loss = jit_train_step(params, x, y)
            loss.block_until_ready()
```

## 3. How to Read an XProf Trace Like a TPU Performance Engineer

When you open the generated `.xplane.pb` trace in TensorBoard (`tensorboard --logdir=/tmp/torchtpu_xprof_logs`), examine these four views in order:

| XProf Tool | What to Look For | Actionable Fix if Suboptimal |
| :--- | :--- | :--- |
| **1. Overview Page** | Check **Step Time Breakdown**: what % of step time is **Idle (Host Stall)** vs. **Compute** vs. **Memory/DMA**? | If Host Idle > 5%, remove per-step `.item()` calls, enable `drop_last=True`, or wrap steps in `jax.lax.scan`. |
| **2. Trace Viewer** | Compare the **Host Python Thread** row against the **TPU TensorCore Ops** row. | In a healthy trace, the Host thread enqueues Step $N+1$ *while* the TPU is still mid-way through Step $N$. Gaps between TPU blocks mean host dispatch bottleneck or shape recompilation. |
| **3. Op Profile** | Ranks HLO ops by % of total TPU time (`fusion`, `dot`, `convolution`, `all-reduce`). Shows **MXU FLOPs %** and **HBM Bandwidth %** per op. | If a `fusion` op takes 30% of step time at 85% HBM BW and 2% FLOPs, it is a candidate for custom **Pallas** kernel fusion (Module 4). |
| **4. Memory Viewer** | Plots peak HBM usage (`GiB`) across the forward and backward pass and lists the largest live buffers. | If near HBM limit (32 GB on `v6e`, 95 GB on `v5p`), enable activation checkpointing (`torch.utils.checkpoint` / `jax.checkpoint`) or shard across chips (Module 5). |

## 4. Guided Hands-On Lab (`labs/lab_02_capture_xprof_trace.py`)

Click **Run Code** in the right-hand pane to execute `labs/lab_02_capture_xprof_trace.py`. It runs a step-annotated Transformer training loop, captures trace files into `/tmp/tpu_lab_xprof`, and inspects the generated `.xplane.pb` / profiler artifact directory structure.
