# Unit 3.1: Roofline Model, Arithmetic Intensity, MFU & LLM Operator Benchmarking

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Derive the **Hardware Roofline Model** (**Arithmetic Intensity** = $\text{FLOPs} / \text{HBM Bytes}$) from first principles to determine whether an operator is **HBM Memory-Bandwidth Bound** or **MXU Compute Bound** on TPU `v6e` (Trillium) and `v5p`.
2. Calculate exact **FLOPs**, **HBM byte traffic**, and **Arithmetic Intensity** for all four core Transformer sub-layers: **QKV / Output Projections**, **Causal SDPA Attention**, **SwiGLU FFN**, and **RMSNorm**.
3. Compute **KV Cache Bytes per Token** across Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) and derive the critical batch size $B_{\text{crit}}$ where LLM autoregressive decoding transitions from memory-bound to compute-bound.
4. Implement synchronized, warmup-isolated benchmarks in both **TorchTPU** and **JAX** that report **Model FLOPs Utilization (MFU %)** and **Achieved HBM Bandwidth (GB/s)**.

## 2. The TPU Roofline Model & Ridge Point Derivation

Every TPU chip is governed by two physical speed limits:
1. **Peak MXU Compute ($\pi_{\text{peak}}$, in TFLOPs/s)**: How fast the systolic array can multiply-accumulate once tiles reside in VMEM.
2. **Peak HBM Bandwidth ($\beta_{\text{HBM}}$, in GB/s)**: How fast the asynchronous DMA engine can stream bytes between on-package HBM and on-chip VMEM.

For an algorithm executing $F$ floating-point operations and transferring $B_{\text{HBM}}$ bytes to/from HBM, its **Arithmetic Intensity ($\mathcal{I}$)** is:

$$\mathcal{I} = \frac{F}{B_{\text{HBM}}} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]$$

Assuming ideal overlap between DMA memory transfers ($T_{\text{mem}} = B_{\text{HBM}} / \beta_{\text{HBM}}$) and MXU execution ($T_{\text{math}} = F / \pi_{\text{peak}}$), the achievable throughput (in FLOPs/s) is bounded by the **Roofline equation**:

$$\text{Achievable FLOPs/s} = \min\left(\pi_{\text{peak}}, \; \mathcal{I} \times \beta_{\text{HBM}}\right)$$

The transition boundary where $T_{\text{math}} = T_{\text{mem}}$ is the chip's **Ridge Point**:

$$\mathcal{I}_{\text{ridge}} = \frac{\pi_{\text{peak}}}{\beta_{\text{HBM}}}$$

| TPU Chip | Peak `bf16` Compute ($\pi_{\text{peak}}$) | Peak HBM Bandwidth ($\beta_{\text{HBM}}$) | Ridge Point ($\mathcal{I}_{\text{ridge}}$) | Practical Matrix Multiply Rule (`bf16`) |
| :--- | :--- | :--- | :--- | :--- |
| **TPU v6e (Trillium)** | `918 TFLOPs/s` | `1,600 GB/s` | **~574 FLOPs / Byte** | A `bf16` matmul $[M, K] \times [K, N]$ needs token batch $M \ge 574$ (or `fp8` weights at $M \ge 287$) to saturate the $256 \times 256$ MXU! |
| **TPU v5p (per chip)** | `459 TFLOPs/s` | `2,765 GB/s` | **~166 FLOPs / Byte** | Higher relative HBM bandwidth; saturates its $128 \times 128$ MXUs at $M \ge 166$. |
| **TPU v5e (per chip)** | `197 TFLOPs/s` | `820 GB/s` | **~240 FLOPs / Byte** | Saturates its MXUs at $M \ge 240$. |

## 3. Transformer Layer Roofline Breakdown: Prefill vs. Decode

Consider a single Transformer layer with batch size $B$, sequence length $S$ ($T = B \cdot S$ total tokens), hidden size $D = 4096$, intermediate FFN size $D_{\text{ff}} = 14336$, and `bfloat16` (`2 bytes` per element):

| Transformer Sub-Layer | Forward FLOPs ($F$) | Unfused HBM Bytes ($B_{\text{HBM}}$) | Arithmetic Intensity ($\mathcal{I}$) | Hardware Regime on `v6e` ($\mathcal{I}_{\text{ridge}} = 574$) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Projection** ($[T, D] \times [D, N]$) | $2 T D N$ | $2(TD + DN + TN)$ | $\approx \frac{T}{1 + T/N + T/D}$ | **Prefill ($T=2048$): Compute-Bound** ($\mathcal{I} \approx 1024$). **Decode ($T=1$): HBM-Bound** ($\mathcal{I} = 1.0$). |
| **SwiGLU FFN** ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) | $6 T D D_{\text{ff}}$ | $6 D D_{\text{ff}} + 8 T D$ | $\approx T$ (for $T \ll D$) | **Prefill: Compute-Bound**. **Decode ($T=1$): HBM-Bound** (dominated by reading $3 \times 2 D D_{\text{ff}}$ weight bytes). |
| **Causal SDPA Attention** (Unfused vs. Pallas Flash) | $4 B S^2 D$ | Unfused: $4 B H S^2 + 8 B S D$; Flash (`splash_sdpa`): $8 B S D$ | Flash: $\frac{S}{2}$ FLOPs/Byte | With Pallas FlashAttention (`splash_sdpa`), $O(S^2)$ attention matrix stays in VMEM; **Compute-Bound** for $S \ge 1024$. |
| **RMSNorm** ($x / \sqrt{\text{mean}(x^2) + \epsilon} \odot \gamma$) | $4 T D$ | $4 T D$ (`read x` + `write y`) | **$1.0$ FLOP / Byte** | **Strictly HBM Bandwidth-Bound at ALL batch sizes**! Must be fused with adjacent residual add. |

## 4. Worked Numerical Walkthrough: KV Cache Sizing & Decode Latency

During autoregressive **Decode**, each step generates 1 token per sequence ($S_{\text{new}} = 1$). For every single generated token, the TPU must stream from HBM:
1. **All Model Weights ($W_{\text{bytes}}$)**: Shared across all $B$ requests in the batch.
2. **The KV Cache ($B \times S_{\text{ctx}} \times \text{KV}_{\text{bytes/token}}$)**: Unique to each sequence in the batch.

### Step 1: KV Cache Bytes per Token Formula (MHA vs. GQA)
For a model with $L$ layers, $n_{\text{kv}}$ Key/Value heads, head dimension $d_{\text{head}}$, and precision $b_{\text{kv}}$ bytes per element (`2` for `bf16`, `1` for `fp8`/`int8`), storing both Key ($K$) and Value ($V$) requires:

$$\text{KV}_{\text{bytes/token}} = 2 \times L \times n_{\text{kv}} \times d_{\text{head}} \times b_{\text{kv}}$$

> **Worked Example — Llama-3-8B (`bf16` on 1× TPU `v6e-1`, 32 GiB HBM, 1,600 GB/s)**:
> * **Architecture**: $P = 8.03 \times 10^9$ params ($W_{\text{bytes}} = 16.06\text{ GB}$ in `bf16`), $L = 32$ layers, $n_{q} = 32$ query heads, **$n_{\text{kv}} = 8$ KV heads (4:1 GQA)**, $d_{\text{head}} = 128$.
> * **KV Cache per Token**: $2 \times 32 \times 8 \times 128 \times 2\text{ B} = \mathbf{131,072\text{ Bytes/token}} \approx \mathbf{128\text{ KiB/token}}$ *(4× smaller than standard MHA's $512\text{ KiB/token}$!)*.
> * **KV Cache for $B = 16$ sequences at context length $S_{\text{ctx}} = 4096$**:
>   $$16 \times 4096 \times 131,072\text{ B} = \mathbf{8.59\text{ GB HBM}}$$
> * **Minimum Theoretical Decode Step Time ($T_{\text{step}}$ at $B=1, S_{\text{ctx}}=4096$)**:
>   $$T_{\text{mem}} = \frac{16.06\text{ GB (weights)} + 0.54\text{ GB (KV cache)}}{1,600\text{ GB/s}} = \mathbf{10.37\text{ ms/token}} \quad (\sim 96\text{ tokens/sec})$$
> * **How to Double Decode Speed on `v6e`**: Quantizing weights to **`fp8_e4m3fn` (`1 byte`)** via `torch._scaled_mm` cuts $W_{\text{bytes}}$ from `16.06 GB` to `8.03 GB`, reducing $T_{\text{mem}}$ from **`10.37 ms` to `5.35 ms` (~187 tokens/sec)**.

## 5. Five Golden Rules of TPU Benchmarking (`go/torch-tpu-benchmarking`)

1. **Always Warm Up First (3–5 iterations)**: Never include Step 0/1 in your timing loop—Step 0 measures XLA graph compilation, not hardware execution.
2. **Synchronize Immediately Before AND After the Timer**:
   * **TorchTPU**: Call `torch.accelerator.synchronize()` immediately before `t0 = time.perf_counter()` and immediately before `t1 = time.perf_counter()` (note: `torch.tpu.synchronize()` is internal-only in PyTorch 2.11+ / `torch_tpu 0.2.0+`).
   * **JAX**: Call `jax.block_until_ready(outputs)` before starting and stopping the timer.
3. **Pass `dynamic=False` to `torch.compile`**: Prevents symbolic shape guards from adding host overhead on fixed-shape benchmarks.
4. **Never Decorate `torch.compile` Benchmarks with `@torch.inference_mode()`**: Always use **`with torch.no_grad():`** instead (inference mode disables tensor version counters required by AOTAutograd).
5. **Report Both Latency and Roofline Utilization**:
   $$\text{TFLOPs/s} = \frac{\text{FLOPs per step}}{\text{Step Time (s)} \times 10^{12}}, \qquad \text{MFU (\%)} = \frac{\text{Achieved TFLOPs/s}}{\pi_{\text{peak}}} \times 100, \qquad \text{BW (GB/s)} = \frac{\text{HBM Bytes}}{\text{Step Time (s)} \times 10^9}$$

## 6. Guided Hands-On Lab (`labs/lab_01_llm_operators_benchmark.py`)

Click **Run Code** in the right-hand pane to execute `labs/lab_01_llm_operators_benchmark.py`, which benchmarks **Prefill Matmul ($M=1024$)**, **Decode Matmul ($M=1$)**, **RMSNorm**, **SwiGLU**, and **Causal SDPA Attention** across both **TorchTPU** and **JAX**, computing Arithmetic Intensity, TFLOPs/s, MFU %, and Achieved HBM Bandwidth (GB/s).
