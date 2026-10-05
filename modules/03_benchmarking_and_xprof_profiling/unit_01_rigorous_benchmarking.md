# Unit 3.1: Roofline Model, Arithmetic Intensity, MFU & LLM Operator Benchmarking

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Apply the **Hardware Roofline Model** (**Arithmetic Intensity** = $\text{FLOPs} / \text{HBM Bytes}$) to determine whether a workload is **HBM Memory-Bandwidth Bound** or **MXU Compute Bound** on TPU `v6e` and `v5p`.
2. Compute **Model FLOPs Utilization (MFU)** and **Achieved HBM Bandwidth (GB/s)** from wall-clock timing measurements.
3. Implement rigorous, bug-free benchmarks comparing **Linear Projections**, **RMSNorm**, **SwiGLU**, and **Scaled Dot-Product Attention (SDPA)** in TorchTPU and JAX.

## 2. The TPU Roofline Model & Arithmetic Intensity

Every TPU chip has two physical speed limits:
1. **Peak MXU Compute ($\pi$, in TFLOPs/s)**: How fast the systolic array can multiply numbers once they are inside VMEM.
2. **Peak HBM Bandwidth ($\beta$, in GB/s)**: How fast the DMA engine can move bytes between on-package HBM and on-chip VMEM.

The ratio of these two hardware limits defines the chip's **Ridge Point**:

$$\text{Ridge Point} = \frac{\text{Peak Compute } \pi}{\text{Peak HBM Bandwidth } \beta} \quad \left[\frac{\text{FLOPs}}{\text{Byte}}\right]$$

| TPU Chip | Peak `bf16` Compute ($\pi$) | Peak HBM Bandwidth ($\beta$) | Ridge Point ($\pi / \beta$) | Practical Implication |
| :--- | :--- | :--- | :--- | :--- |
| **TPU v6e (Trillium)** | `918 TFLOPs/s` | `1,640 GB/s` | **~560 FLOPs / Byte** | A `bf16` matmul needs batch/token dimension $M \ge 560$ to saturate the $256 \times 256$ MXU! |
| **TPU v5p (per chip)** | `459 TFLOPs/s` | `2,765 GB/s` | **~166 FLOPs / Byte** | Higher relative HBM bandwidth; saturates MXU at $M \ge 166$. |
| **TPU v5e (per chip)** | `197 TFLOPs/s` | `820 GB/s` | **~240 FLOPs / Byte** | Saturates MXU at $M \ge 240$. |

### Why LLM Prefill Is Compute-Bound While LLM Decode Is Memory-Bound
Consider multiplying token activations $X$ of shape `[Tokens=M, K=4096]` by weight matrix $W$ of shape `[K=4096, N=4096]` in `bfloat16` (2 bytes per element):
* **Total FLOPs**: $2 \cdot M \cdot K \cdot N$
* **Total HBM Bytes Read/Written**: $2 \cdot (M K + K N + M N)$ bytes.
* **Case 1: LLM Prefill / Training ($M = 2048$ tokens)**:
  $$\text{Arithmetic Intensity} = \frac{2 \cdot 2048 \cdot 4096^2}{2 \cdot (2048 \cdot 4096 + 4096^2 + 2048 \cdot 4096)} \approx \mathbf{1024 \text{ FLOPs/Byte}} > 560$$
  **Compute-Bound (MXU Bound)**: Measured by **TFLOPs/s** and **MFU (%)**.
* **Case 2: LLM Autoregressive Decode ($M = 1$ token)**:
  $$\text{Arithmetic Intensity} = \frac{2 \cdot 1 \cdot 4096^2}{2 \cdot 4096^2} = \mathbf{1.0 \text{ FLOP/Byte}} \ll 560$$
  **Memory-Bandwidth Bound (HBM Bound)**: The MXU sits 99.8% idle waiting to stream the 32 MB weight matrix from HBM! Measured by **Achieved HBM Bandwidth (GB/s)**.

## 3. Five Golden Rules of TPU Benchmarking (`go/torch-tpu-benchmarking`)

1. **Always Warm Up First (3–5 iterations)**: Never include Step 0/1 in your timing loop—Step 0 measures XLA graph compilation, not hardware execution.
2. **Synchronize Before AND After the Timer**:
   * **TorchTPU**: Call `torch.accelerator.synchronize()` (or `torch.tpu.synchronize()`) immediately before `t0 = time.perf_counter()` and immediately before `t1 = time.perf_counter()`.
   * **JAX**: Call `jax.block_until_ready(outputs)` before starting and stopping the timer.
3. **Use `dynamic=False` on `torch.compile`**: Prevents symbolic shape guards from adding host overhead.
4. **Never Decorate `torch.compile` Benchmarks with `@torch.inference_mode()`**: Always use **`with torch.no_grad():`** instead (inference mode disables tensor version counters needed by AOTAutograd).
5. **Report Both Median/Mean Latency and Hardware Utilization**:
   * $\text{TFLOPs/s} = \frac{\text{FLOPs per step}}{\text{Step Time (s)} \times 10^{12}}$
   * $\text{MFU (\%)} = \frac{\text{Achieved TFLOPs/s}}{\text{Chip Peak TFLOPs/s}} \times 100$
   * $\text{Achieved HBM BW (GB/s)} = \frac{\text{Bytes Read + Written}}{\text{Step Time (s)} \times 10^9}$

## 4. Guided Hands-On Lab (`labs/lab_01_llm_operators_benchmark.py`)

Click **Run Code** in the right-hand pane to execute `labs/lab_01_llm_operators_benchmark.py`, which benchmarks **Prefill Matmul ($M=1024$)**, **Decode Matmul ($M=1$)**, **RMSNorm**, **SwiGLU**, and **Scaled Dot-Product Attention (SDPA)** across TorchTPU and JAX, computing Arithmetic Intensity, TFLOPs/s, and Achieved HBM Bandwidth (GB/s) for each operator.
