# Unit 6.3: Cloud TPU Consulting Cookbook — `v5p` vs. `v6e` Sizing Matrix & Production Recipes

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Use the verified hardware constants from `torch_tpu/_internal/utils/hardware.py` (`v5e`, `v5p`, `v6e`, `v7`) to recommend the optimal Cloud TPU generation for any customer training or inference workload.
2. Apply the **Roofline Ridge-Point Rule** to explain why **TPU `v6e` (Trillium)** wins on compute-bound prefill, vision/diffusion, and LoRA/SFT with $2\times$ BF16 TFLOP/s per chip (`918` vs. `459`), whereas **TPU `v5p`** wins on memory-capacity-bound large models (`95 GiB` vs. `32 GiB` HBM per chip) and memory-bandwidth-bound autoregressive decode (`2,765 GB/s` vs. `1,600 GB/s`).
3. Use the ready-to-run consulting recipes stored in `/usr/local/google/home/asrarfarooq/tpu-learning-lab/recipes/` (and surfaced directly in the **Recipes** tab in the left sidebar) for environment tuning, Hugging Face benchmarking, FP8 `torch._scaled_mm` quantization, Ragged Mixture-of-Experts (`ragged_dot`), and `SimpleFSDP` / `tpu_dist` distributed training.

## 2. Verified Hardware Reference Table (`torch_tpu/_internal/utils/hardware.py`)

Every constant below is verified directly against `//depot/google3/third_party/py/torch_tpu/_internal/utils/hardware.py`:

| TPU Generation | `TpuVersion` | Exposed Devices / Chip | HBM Capacity / Chip | Peak Dense BF16 TFLOP/s / Chip | HBM Bandwidth / Chip | MXU Dimensions | Single-Host `TORCH_TPU_TOPOLOGY` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TPU v5e** | `TPU v5e` (`0x0063`) | `1` | **`16 GiB`** | **`197 TFLOP/s`** | `819 GB/s` | $128 \times 128$ (4 MXUs) | `1: "1,1,1"`, `4: "2,2,1"`, `8: "2,4,1"` |
| **TPU v5p** | `TPU v5p` (`0x0062`) | `1` (Megacore) | **`95 GiB`** | **`459 TFLOP/s`** | `2,765 GB/s` | $128 \times 128$ (8 MXUs) | `1: "1,1,1"`, `2: "1,2,1"`, `4: "2,2,1"`, `8: "2,2,2"` |
| **TPU v6e (Trillium)** | `TPU v6e` (`0x006f`) | `1` | **`32 GiB`** | **`918 TFLOP/s`** | `1,600 GB/s` | $256 \times 256$ (2 MXUs) | `1: "1,1,1"`, `4: "2,2,1"`, `8: "2,4,1"` |
| **TPU v7 (Ironwood)** | `TPU v7` (`0x0076`) | `2` (`96 GiB` each) | **`192 GiB`** | **`2,307 TFLOP/s`** (`1,153.5` / dev) | `7,370 GB/s` | $256 \times 256$ | `2: "1,1,1,2"`, `4: "1,2,1,2"`, `8: "2,2,1,2"` |

## 3. Consulting Decision Matrix: When to Choose `v6e` (Trillium) vs. `v5p`

Because **`v6e` has $2.0\times$ the BF16 compute per chip (`918` vs. `459` TFLOP/s)** while **`v5p` has $2.97\times$ the HBM capacity (`95 GiB` vs. `32 GiB`) and $1.73\times$ the HBM bandwidth (`2,765` vs. `1,600 GB/s`)**, their performance crossover depends directly on the workload's **Arithmetic Intensity** and **Memory Footprint**:

| Workload Archetype | Dominant Bottleneck | Recommended TPU | Technical Rationale |
| :--- | :--- | :--- | :--- |
| **LLM Prefill / Prompt Processing (`seq_len >= 512`)** | MXU Compute (`FLOP/s`) | **`v6e` (Trillium)** | Arithmetic intensity is $> 512$ FLOPs/byte (well above `v6e`'s `574` ridge point). `v6e`'s $256 \times 256$ MXUs deliver up to **2.0x higher prefill throughput per chip**. |
| **Small/Medium LLM SFT & LoRA (`0.5B` – `8B` models)** | MXU Compute + HBM Capacity | **`v6e-1` / `v6e-4` / `v6e-8`** | With `bfloat16` weights + LoRA (or `SimpleFSDP` across 4–8 chips), weights fit comfortably in `32 GiB`/chip and `v6e` completes training steps **1.6x–1.9x faster per chip**. |
| **Low-Batch Autoregressive Decode (`batch <= 16`)** | HBM Bandwidth (`GB/s`) | **`v5p` (or `v6e` with FP8/Int8 weight quantization)** | Unquantized decode at `batch=1` has arithmetic intensity $\approx 1$ FLOP/byte, making it 100% HBM bandwidth bound (`v5p` `2,765 GB/s` > `v6e` `1,600 GB/s`). Quantizing weights to `float8_e4m3fn` (`torch._scaled_mm`) on `v6e` halves HBM traffic and restores `v6e` cost-efficiency. |
| **Large-Scale 3D Parallel Pre-Training (`30B` – `405B+` models)** | HBM Capacity (`95 GiB`/chip) & 3D Torus ICI | **`v5p` (`v5p-8` to `v5p-8960+`)** | `v5p` provides **`95 GiB` HBM per chip** (`380 GiB` per 4-chip VM) and a **3D Torus ICI interconnect** across thousands of chips, minimizing tensor-parallel splits. |

## 4. Advanced TorchTPU Production Primitives: FP8 `_scaled_mm`, Ragged MoE & `topology_aware_mesh`

### A. Native FP8 & Blockwise Scaled Matmul (`torch._scaled_mm`)
TorchTPU lowers `torch._scaled_mm` (`csrc/ops/matmul/scaled_mm_aten_kernels.cc`) directly to XLA `DotGeneral` on `torch.float8_e4m3fn` (`MAX = 448.0`) and `torch.float8_e5m2` (`MAX = 57344.0`) with three scaling granularities:
* **TensorWise**: `scale_a` and `scale_b` are scalar `float32` tensors (`numel() == 1`).
* **RowWise**: `scale_a` has shape `(M, 1)` and `scale_b` has shape `(1, N)`.
* **BlockWise**: `scale_a` has shape `(M, K // block_size)` and `scale_b` has shape `(K // block_size, N)`.

```python
fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
scale_x = (x_bf16.abs().amax() / fp8_max).float().clamp(min=1e-12)
scale_w = (w_bf16.abs().amax() / fp8_max).float().clamp(min=1e-12)

x_fp8 = (x_bf16.float() / scale_x).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
w_fp8_col_major = (w_bf16.float() / scale_w).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).t().contiguous().t()

# Executes in FP8 on TPU MXU and returns bfloat16 output!
out_bf16 = torch._scaled_mm(x_fp8, w_fp8_col_major, scale_a=scale_x, scale_b=scale_w, out_dtype=torch.bfloat16)
```

### B. Zero-Break Mixture-of-Experts with `torch.tpu.ops.ragged_dot`
Standard Hugging Face MoE implementations (`Qwen3-MoE`, `Gemma-4`, `Mixtral`) loop over `num_experts` and call `torch.where(top_k_index == expert_idx)` (`aten.nonzero`), which creates dynamic data-dependent shapes and breaks `torch.compile`. TorchTPU solves this with `torch.tpu.ops.ragged_dot`:
1. Sort routed tokens by expert assignment via `torch.argsort(expert_ids, stable=True)`.
2. Compute the `[num_experts]` token-count histogram on device without host sync using `group_sizes.scatter_add_(0, expert_ids, torch.ones_like(expert_ids))`.
3. Execute all experts in **one single XLA `ragged_dot` kernel**:

```python
from torch.tpu.ops import ragged_dot

# lhs: [Total_Routed_Tokens, Hidden], rhs: [Num_Experts, Hidden, Intermediate], group_sizes: [Num_Experts]
expert_out = ragged_dot(lhs=sorted_tokens, rhs=expert_weights, group_sizes=group_sizes)
```

### C. Multi-Slice Hybrid ICI + DCN Mesh (`torch.tpu.topology_aware_mesh`)
When scaling across multiple TPU slices over Data Center Network (DCN), always construct the PyTorch `DeviceMesh` via `torch.tpu.topology_aware_mesh`:

```python
mesh = torch.tpu.topology_aware_mesh(
    mesh_shape=(num_slices, chips_per_slice),
    mesh_dim_names=("ddp_dcn", "fsdp_ici"),
)
```

## 5. Production Recipes Library (`/recipes/` & Sidebar `Recipes` Tab)

Switch to the **Recipes** tab in the left sidebar to inspect and run all 5 standalone production scripts in `/usr/local/google/home/asrarfarooq/tpu-learning-lab/recipes/`:

1. **`recipes/01_env_and_cache_setup.sh`**: Sets `TORCH_TPU_TIER2_COMPILATION_CACHE`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT`, `LIBTPU_INIT_ARGS`, and `python -m torch.tpu.distributed`.
2. **`recipes/02_hf_llm_prefill_decode_benchmark.py`**: Hugging Face `AutoModelForCausalLM` (`Qwen3`, `Llama-3.2`, `Gemma-2`) benchmark with `StaticCache`, 4D causal mask, `SDPBackend.OVERRIDEABLE`, and `torch.tpu._get_cache_misses()` verification.
3. **`recipes/03_single_trace_and_lora_sft_recipe.py`**: SFT / LoRA training template combining `SingleTraceTrainer` (`fullgraph=True`), `patch_optimizer_graph_breaks()`, and chunkwise vocabulary cross-entropy loss.
4. **`recipes/04_fp8_scaled_mm_and_quant_benchmark.py`**: Benchmarks `torch._scaled_mm` (`float8_e4m3fn` TensorWise and RowWise scaling) vs. `bfloat16` and `torch.tpu.Precision` modes.
5. **`recipes/05_ragged_moe_qwen_gemma_benchmark.py`**: Benchmarks zero-graph-break Mixture-of-Experts routing (`torch.tpu.ops.ragged_dot` + `scatter_add_`) against naive expert loops.

## 6. Guided Hands-On Lab (`labs/lab_03_v5p_vs_v6e_recipe_advisor.py`)

Run `labs/lab_03_v5p_vs_v6e_recipe_advisor.py` in the right-hand lab editor to calculate exact HBM capacity, minimum chip topology (`v6e-1`, `v6e-4`, `v6e-8`, `v5p-8`), and Roofline latency across `Qwen3-0.6B`, `Llama-3.2-1B`, `Gemma-2-2B`, `Llama-3.1-8B`, `Qwen3-30B-A3B`, and `Llama-3.1-70B`.
