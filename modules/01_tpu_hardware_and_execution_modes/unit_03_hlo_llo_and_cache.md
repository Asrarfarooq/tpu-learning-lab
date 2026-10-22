# Unit 1.3: Reading XLA HLO / LLO Dumps, Compilation Cache & Dynamic Shape Traps

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Dump and read **XLA HLO (High-Level Optimizer)** text graphs (`--xla_dump_to`) to verify whether XLA fused your operators or inserted expensive `copy` / `reshape` layout conversions.
2. Configure the **Persistent Compilation Cache** in both TorchTPU (`TORCH_TPU_TIER2_COMPILATION_CACHE`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`) and JAX (`jax_compilation_cache_dir`) to eliminate cold-start compilation latency across process restarts.
3. Diagnose and eliminate **Dynamic Shape Recompilation Traps** (varying batch sizes, variable sequence lengths, and boolean masking).

## 2. How to Read an XLA HLO Dump

When you enable the **Dump XLA HLO** checkbox in the right-hand lab runner (or set `XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"`), the XLA compiler writes human-readable snapshots of your computation graph before and after optimization:

* `module_0000...before_optimizations.txt`: The raw unoptimized graph emitted by TorchTPU or JAX. Every `+`, `*`, and `tanh` appears as a separate instruction.
* `module_0000...after_optimizations.txt`: The final optimized HLO graph after **Operator Fusion**, **Layout Assignment**, **Algebraic Simplification**, and **Dead Code Elimination**.

### Anatomy of an Optimized HLO Graph

```text
HloModule jit_fused_block

%fused_computation (p0: bf16[1024,1024], p1: bf16[1024,1024])
    -> bf16[1024,1024] {
  %p0 = bf16[1024,1024]{1,0} parameter(0)
  %p1 = bf16[1024,1024]{1,0} parameter(1)
  %add = bf16[1024,1024]{1,0} add(%p0, %p1)
  %cst = bf16[] constant(0.5)
  %bcast = bf16[1024,1024]{1,0} broadcast(%cst), dimensions={}
  ROOT %mul = bf16[1024,1024]{1,0} multiply(%add, %bcast)
}

ENTRY %main (A0: bf16[1024,1024], A1: bf16[1024,1024])
    -> bf16[1024,1024] {
  %A0 = bf16[1024,1024]{1,0} parameter(0)
  %A1 = bf16[1024,1024]{1,0} parameter(1)
  %dot = bf16[1024,1024]{1,0} dot(%A0, %A1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT %fusion = bf16[1024,1024]{1,0} fusion(%dot, %A0),
      kind=kLoop, calls=%fused_computation
}
```

### Key HLO Symbols to Look For
1. **`dot(..., lhs_contracting_dims={1}, rhs_contracting_dims={0})`**: Executes on the **TPU MXU (Matrix Multiply Unit)**.
2. **`fusion(..., kind=kLoop, calls=%fused_computation)`**: Proves that XLA fused all instructions inside `%fused_computation` into a **single VMEM pass** with zero intermediate HBM reads/writes.
3. **`{1,0}` Layout Annotation**: Indicates minor-to-major physical layout in memory (dimension `1` is contiguous in memory, i.e., row-major). On TPUs, XLA also appends tiling metadata such as `{1,0:T(8,128)(2,1)}` indicating $(8 \times 128)$ sub-tiles packed for `bfloat16`.
4. **Warning Signs in HLO**:
   * Unexpected **`copy`** instructions between compute ops mean XLA had to transpose or re-layout a tensor in HBM.
   * Multiple separate `kLoop` fusions that were broken apart by an intermediate host callback or unsupported op.

## 3. The 3-Tier Compilation Cache (Cold Start vs. Warm Hit)

Compiling a large Transformer graph with XLA takes anywhere from seconds to minutes because XLA solves a global scheduling and VMEM allocation problem. To avoid recompiling the same graph:

| Cache Tier | Scope | TorchTPU Configuration | JAX Configuration |
| :--- | :--- | :--- | :--- |
| **Tier 1: In-Memory** | Current Python process | Automatic (`step >= 2` hits memory cache in microseconds) | Automatic (`@jax.jit` in-memory dispatch table) |
| **Tier 2: Shared Memory Cache** | Across processes on the same host (`/dev/shm/torch_tpu_cache/<name>`) | `os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "my_shared_cache"` | Covered by persistent cache dir |
| **Tier 3: Persistent Disk / GCS Cache** | Across VM restarts or multi-host clusters (local disk or `gs://` bucket) | `os.environ["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"] = "/tmp/tpu_compilation_cache"` | `jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")` |

## 4. Dynamic Shape Recompilation Traps & How to Fix Them

XLA binaries are **statically specialized** to the exact `shape` and `dtype` of every input argument.

### Trap 1: Variable Batch Size on the Final Batch of an Epoch
If your dataset has `1,000` examples and `batch_size = 64`:
* Steps 0 to 14 have shape `(64, 512)` -> Compiled once on Step 0, fast on Steps 1–14.
* Step 15 (the last batch) has remainder shape `(40, 512)` -> **Triggers a full XLA recompilation just for 1 step!**
* **Fix**: Always set `drop_last=True` on your PyTorch `DataLoader` (or pad the final batch with zeros and mask the loss).

### Trap 2: Variable Sequence Lengths in LLM Prompts
Passing unpadded prompts of length `17`, `43`, `89`, `102` causes XLA to recompile on every single request.
* **Fix**: Pad inputs to a small set of **power-of-two buckets** (`128`, `256`, `512`, `1024`, `2048`) so XLA compiles at most 5 bucket shapes and hits the cache 100% of the time thereafter.

### Programmatic Recompilation Guard in TorchTPU (`torch.tpu._get_cache_misses`)
In TorchTPU benchmarks (`torch_tpu/perflab/`), always assert zero unexpected recompilations during steady-state iterations by sampling `torch.tpu._get_cache_misses()` before and after your timed loop:

```python
misses_before = torch.tpu._get_cache_misses()
for batch in steady_state_batches:
    loss = compiled_step(batch)
torch.accelerator.synchronize()
misses_after = torch.tpu._get_cache_misses()
assert misses_after == misses_before, f"Recompilation storm: {misses_after - misses_before} new compiles!"
```

## 5. Bounded Dynamism in TorchTPU (`docs/dynamic_shapes.md`)

What happens when tensor shapes *do* vary at runtime (e.g., variable prompt lengths or growing KV caches)? Unlike GPUs where kernels launch dynamically with grid dimensions passed as scalar registers, XLA requires bounded compile-time upper bounds for VMEM tile allocation. TorchTPU implements **Bounded Dynamism**:

| `torch.compile` Mode | Behavior on Shape Changes | How TorchTPU Compiles It |
| :--- | :--- | :--- |
| **`dynamic=False`** *(Recommended for fixed buckets)* | Compiles a specialized static XLA binary for each distinct input shape. | Zero dynamic-dimension overhead; highest MXU/VMEM efficiency. Pad inputs to fixed power-of-2 or 128-aligned buckets on host. |
| **`dynamic=None`** *(Default auto-dynamic)* | Compiles Step 1 statically. If a dimension changes on Step 2, TorchDynamo automatically marks that dimension symbolic (`SymInt`). | TorchTPU rounds the dynamic dimension up to a **128-aligned upper bound (`bucket_size`)**, pads the tensor on device, and emits `stablehlo.set_dimension_size` / `stablehlo.get_dimension_size` so reductions (`mean`, `softmax`) mask out padded elements automatically. |
| **`dynamic=True` or `torch._dynamo.mark_dynamic(x, dim)`** | Immediately traces the specified dimension(s) symbolically from Step 1. | Allocates up to **`MAX_BUCKETS_PER_DIM = 8`** 128-aligned buckets per dynamic dimension; exceeding 8 bucket expansions raises an error to prevent unbounded HBM fragmentation. |

### The 3-Step Autoregressive Decode Warmup Pattern (`docs/dynamic_shapes.md`)
When using `torch.compile(model)` (`dynamic=None`) on a loop where sequence length grows (or marking a dimension explicitly with `torch._dynamo.mark_dynamic(input_ids, 1)`), understand the **3-step warmup lifecycle**:
1. **Step 1 (`seq_len = 10`)**: TorchDynamo compiles a static graph for `seq_len = 10`.
2. **Step 2 (`seq_len = 11`)**: TorchDynamo detects that `dim=1` changed, generalizes `dim=1` to a symbolic `SymInt`, and TorchTPU compiles a **Bounded Dynamic StableHLO graph** bounded at `bucket_size = 128` using `stablehlo.set_dimension_size`.
3. **Steps 3..128 (`seq_len = 12..128`)**: Every step executes inside the existing `[B, <=128]` bounded XLA binary with **zero recompilations**!

```python
import torch

# Explicit dynamic dimension marking before Step 1 (skips the Step 1 -> Step 2 specialization recompile):
torch._dynamo.mark_dynamic(x, 1)
out = compiled_model(x)
```

## 6. Guided Hands-On Lab (`labs/lab_03_inspect_hlo_and_recompiles.py`)

1. Check the **Dump HLO** checkbox in the right-hand lab toolbar.
2. Click **Run** to execute `labs/lab_03_inspect_hlo_and_recompiles.py`, which runs both **PyTorch (`torch.compile`)** and **JAX (`@jax.jit`)** side-by-side.
3. In the Output tab, observe the **latency spike when input shapes change dynamically** vs. **power-of-two bucketed static shapes**, and inspect the lowered StableHLO / post-optimization HLO fusion instructions.
