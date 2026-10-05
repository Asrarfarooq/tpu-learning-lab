# Unit 6.1: Hugging Face LLMs on TorchTPU — 4D Causal Masks, `StaticCache` & Splash Attention

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Load open-weight Hugging Face `transformers` models (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `meta-llama/Llama-3.2-1B`, `google/gemma-2-2b`, `google/gemma-3-270m`) directly onto `torch.device("tpu")` in `bfloat16` without host-memory bloat.
2. Eliminate `transformers.masking_utils` graph breaks under `torch.compile(backend="tpu", dynamic=False)` using either unpadded mask removal or precomputed **4D Causal Masks**.
3. Implement zero-recompilation autoregressive decoding using **`transformers.cache_utils.StaticCache`**, eager prefill, compiled decode steps, and in-place warmup cache rewinding (`torch_tpu/perflab/steps/decode.py`).
4. Accelerate Sliding Window and Grouped-Query Attention (GQA) using TorchTPU's Pallas **`torch.tpu.ops.splash_sdpa`** kernel and **`SDPBackend.OVERRIDEABLE`**.

## 2. Direct-to-TPU Model Initialization & The `@torch.no_grad()` Rule

When instantiating a multi-billion parameter Hugging Face model on a Cloud TPU VM, allocating weights on the Host CPU first and calling `.to("tpu")` afterwards doubles host DRAM pressure and triggers slow PCIe copies. Instead, allocate directly inside a `with torch.device("tpu"):` context or pass `torch_dtype=torch.bfloat16`:

```python
import torch
import torch_tpu  # Registers the 'tpu' PrivateUse1 accelerator backend
import transformers

device = torch.device("tpu")

# Allocate model parameters directly in bfloat16 on TPU HBM
with torch.device("tpu"):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        use_cache=False,  # False for training/prefill-only; True with StaticCache for decode
    ).eval()
```

> **Caution — Never Use `@torch.inference_mode()` with `torch.compile(backend="tpu")`:**
> In `torch_tpu` (`docs/markdown/error_glossary.md`), `torch.inference_mode()` disables tensor version counters required by `torch.embedding` and AOTAutograd/TorchDynamo graph capture, raising `RuntimeError: Cannot set version_counter for inference tensor`. **Always use `with torch.no_grad():` for evaluation and inference on TorchTPU.**

## 3. Eliminating Hugging Face `masking_utils` Graph Breaks

When you pass a standard 2D `attention_mask` (shape `[batch_size, seq_len]`) from `AutoTokenizer` into `model(**inputs)`, `transformers.masking_utils` executes Python control-flow heuristics (such as `_ignore_causal_mask_sdpa` and `find_packed_sequence_indices`) that call `.all()` or `.item()` on the mask tensor. On a TPU, inspecting tensor values from Python forces a synchronous device-to-host transfer and causes a **TorchDynamo graph break**.

```
Standard 2D attention_mask Path (Causes Graph Break & Host Sync!):
  AutoTokenizer -> 2D attention_mask [B, S]
    -> transformers.masking_utils calls mask.all() / .item()
    -> Host CPU blocks waiting for TPU -> Graph Break!

TorchTPU Production Fix A (Full-Length Unpadded Batches):
  inputs.pop("attention_mask", None)
    -> Model uses pure internal causal mask (is_causal=True)
    -> 0 Host Syncs, 0 Graph Breaks!

TorchTPU Production Fix B (Padded / Sliding-Window Batches):
  Precompute 4D Causal Mask [B, 1, S, S] outside compiled region
    -> transformers.masking_utils takes early return on 4D mask
    -> 0 Host Syncs, 0 Graph Breaks!
```

### Verified Pattern (`torch_tpu/perflab/model_helpers.py` & `model_utils.py`)

```python
from transformers import masking_utils

# Fix A: When benchmarking or training on fixed-length packed sequences without padding:
inputs.pop("attention_mask", None)

# Fix B: When a mask is required, precompute the 4D mask [B, 1, S, S] before calling the compiled model:
cache_position = torch.arange(seq_len, device=device)
causal_mask_4d = masking_utils.create_causal_mask(
    config=model.config,
    input_embeds=dummy_embeds,  # shape [B, S, D], dtype=torch.bfloat16
    attention_mask=attention_mask_2d,
    cache_position=cache_position,
    past_key_values=None,
)
```

## 4. Zero-Recompilation Decoding with `StaticCache` (`perflab/steps/decode.py`)

By default, Hugging Face `model.generate()` uses `DynamicCache`, which appends each newly generated token's Key and Value tensors along the sequence dimension (`S`, `S+1`, `S+2`, ...). Because XLA compiles binaries for static tensor shapes, growing the KV cache shape by 1 on every token triggers a full XLA recompilation on **every single generated token**.

To run autoregressive decoding at full TPU memory bandwidth with **zero post-warmup recompilations**, follow the exact 4-stage pipeline from `torch_tpu/perflab/steps/decode.py`:

| Stage | Execution Mode | Operation & Key TPU Engineering Detail |
| :--- | :--- | :--- |
| **1. Allocate `StaticCache`** | Setup | Pre-allocate fixed-size KV buffers `[B, num_kv_heads, max_cache_len, head_dim]` in TPU HBM (`cache_utils.StaticCache`). |
| **2. Prefill Prompt** | Eager (`torch.no_grad()`) | Run the full prompt `[B, prompt_len]` once to populate `keys[:, :, :prompt_len, :]` and extract the first generated token with **`keepdim=True`**. |
| **3. Warmup & Rewind Cache** | Compiled (`dynamic=False`) | Run 2 warmup decode steps to compile the single-token decode graph (`input_ids` shape `[B, 1]`), then **zero out the decode slice in-place** (`layer.keys[:, :, prompt_len:, :].zero_()`) without reallocating the cache! |
| **4. Measured Decode Loop** | Compiled (`dynamic=False`) | Step `cache_position` (`torch.tensor([pos], device=device)`) inside the compiled model with `torch.tpu._get_cache_misses()` remaining strictly constant (`0` new compiles). |

> **Important — Why `keepdim=True` Matters on `argmax` (`perflab/steps/decode.py`):**
> Always compute `next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)` instead of `torch.argmax(..., dim=-1).unsqueeze(-1)`. On TorchTPU, `.unsqueeze(-1)` creates a lazy `as_strided` view tensor, which forces the runtime to launch an extra eager `as_strided` XLA kernel before entering your compiled decode step.

## 5. Attention Backends on TorchTPU: `SDPBackend` & Pallas `splash_sdpa`

TorchTPU provides three high-performance paths for `F.scaled_dot_product_attention` (`csrc/ops/scaled_dot_product_attention/scaled_dot_product_attention_aten_kernels.cc`):

1. **`SDPBackend.FLASH_ATTENTION` (Mosaic FlashAttention)**: Automatically selected when shapes are static, `dtype` is `bfloat16` or `float32`, `dropout_p == 0.0`, and `head_dim < 128` or `head_dim % 128 == 0`.
2. **`SDPBackend.OVERRIDEABLE` (StableHLO SDPA)**: Used across `torch_tpu` Hugging Face benchmarks (`with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.OVERRIDEABLE]):`) to lower SDPA directly into XLA's native fused dot-product attention compiler pass.
3. **`torch.tpu.ops.splash_sdpa` (Pallas Splash Attention for GQA & Sliding Window)**:
   Standard PyTorch Sliding Window Attention (used in **Gemma-2 / Gemma-3 / Gemma-4** and **Mistral/Qwen**) materializes an $O(S^2)$ boolean mask in HBM and calls `torch.repeat_interleave` to duplicate KV heads for Grouped-Query Attention (GQA). TorchTPU's Pallas `splash_sdpa` kernel eliminates the 2D mask from HBM entirely (enforcing the causal/sliding window via SRAM block coordinates) and broadcasts GQA heads on-the-fly inside the TPU VPU:

```python
from torch.tpu.ops import splash_sdpa

# Drop-in replacement for GQA + Sliding Window Attention on TPU:
attn_out = splash_sdpa(
    q, k, v,
    scale=head_dim ** -0.5,
    is_causal=True,
    local_window_size=sliding_window - 1,  # Or None for full causal attention
    enable_gqa=(num_q_heads != num_kv_heads),
    block_q=1024,
    block_kv=1024,
    block_dkv=1024,
)
```

In internal Gemma-4 benchmarks (`google/examples/gemma4/gemma4_tutorial.md`, `batch=4, seq_len=1024`), switching from eager masked SDPA to `splash_sdpa` reduced forward latency from **`33.89 ms` to `25.21 ms` (1.34x faster)** and cut forward peak HBM from **`12.84 GB` to `5.20 GB` (2.47x memory reduction)**.

## 6. Guided Hands-On Lab (`labs/lab_01_hf_static_cache_decode_benchmark.py`)

In the right-hand lab editor, examine `labs/lab_01_hf_static_cache_decode_benchmark.py`:
1. It builds a Grouped-Query Attention (GQA) Transformer decoder matching the architecture of **Qwen3 / Llama-3.2 / Gemma** (using `transformers` when installed, or a self-contained GQA + RoPE + SwiGLU decoder with identical tensor semantics).
2. It compares **Dynamic KV Cache allocation** (which changes tensor shapes on every token `[B, H, S+t, D]`) against a **Pre-Allocated `StaticCache` buffer** (`[B, H, max_cache_len, D]`) updated via static `cache_position` indexing.
3. It verifies `keepdim=True` token selection, in-place cache rewinding, and computes **Prefill Time-To-First-Token (TTFT)**, **Decode Inter-Token Latency (ITL)**, and **HBM Bandwidth Utilization (%)** on `v6e-1` vs. `v5p`.
