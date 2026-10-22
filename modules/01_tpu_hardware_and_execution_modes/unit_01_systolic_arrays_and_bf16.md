# Unit 1.1: Systolic Arrays (MXU), `bfloat16` Bit Layout & Multi-Pass Precision

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Explain how the TPU **Matrix Multiply Unit (MXU)** executes matrix multiplications using a 2D **Systolic Array** ($128 \times 128$ on v5e/v5p, $256 \times 256$ on v6e Trillium).
2. Compare the IEEE-754 bit layouts of **`float32`**, **`float16` (FP16)**, and **`bfloat16` (BF16)**, and explain why Google invented `bfloat16` specifically for TPUs.
3. Control TPU matrix multiplication precision in both **TorchTPU** (`torch.set_float32_matmul_precision`, `with torch.tpu.precision(torch.tpu.Precision.HIGHEST):`) and **JAX** (`jax.lax.Precision.DEFAULT`, `HIGH`, `HIGHEST`) and measure the throughput vs. numerical accuracy trade-off.

## 2. Inside the TPU TensorCore: VPU vs. MXU (Systolic Array)

A CPU or GPU core is an instruction-fetching processor: for every fused multiply-add (`FMA`), it fetches instructions, decodes them, reads operands from a register file, writes results back to the register file, and arbitrates cache access.

Google's TPU **Matrix Multiply Unit (MXU)** eliminates register-file read/write overhead during matrix multiplication by using a **2D Systolic Array**:

```schematic
mxu-systolic-array
```

### How the Systolic Array Achieves Massive Density
1. **No Register File Access Between Steps**: Activation values ($X$) flow horizontally from left to right across the grid while partial sums flow vertically from top to bottom, passing directly from one Arithmetic Logic Unit (ALU) cell to its neighbor on every clock tick.
2. **Hardware Dimensions**:
   * **TPU v5e / v5p**: Each MXU is a **$128 \times 128$** grid of multiply-accumulate cells ($16,384$ multiply-accumulates per clock cycle per MXU).
   * **TPU v6e (Trillium)**: Each MXU is doubled along both dimensions to a **$256 \times 256$** grid ($65,536$ multiply-accumulates per clock cycle—**4x the compute density per MXU**).
3. **Tile Alignment Rule**: Because the physical array processes $128 \times 128$ (or $256 \times 256$) tiles as an indivisible unit, matrix dimensions ($M, K, N$) that are multiples of **128** (or **256** on `v6e`) achieve near-100% cell utilization. Non-multiple dimensions (such as $M=1000$) must be zero-padded by XLA to the next tile boundary ($1024$).

## 3. Floating-Point Bit Layouts: Why TPUs Use `bfloat16` & `fp8`

Every floating-point number is stored in binary scientific notation:

$$\text{Value} = (-1)^{\text{sign}} \times 2^{(\text{exponent} - \text{bias})} \times \left(1 + \frac{\text{mantissa}}{2^M}\right)$$

* **Exponent bits** determine the **dynamic range** (how large or small a number can be before overflowing to `Inf` or underflowing to `0.0`, which destroys tiny neural network gradients).
* **Mantissa (fraction) bits** determine the **fine-grained precision** (how many significant decimal digits are preserved).

```schematic
floating-point-bit-layouts
```

### Why the MXU Multiplies in `bfloat16` / `fp8` and Accumulates in `float32`
The physical area of a hardware multiplier circuit scales **quadratically** with the number of mantissa bits:
* A `float32` multiplier requires $(23+1)^2 = 576$ unit gates.
* A `bfloat16` multiplier requires $(7+1)^2 = 64$ unit gates (**9x smaller in silicon area!**).
* Inside the TPU MXU, each cell multiplies **`bfloat16`** (or **`float8_e4m3fn` / `float8_e5m2`** via `torch._scaled_mm`) inputs and adds the product into a **`float32` accumulator register**, preserving numerical stability across thousands of summed terms.

## 4. Controlling `float32` Matmul Precision (1-Pass vs. 3-Pass vs. 6-Pass)

What happens when you pass two **`float32`** tensors `A` and `B` into `A @ B` on a TPU whose physical MXU multipliers only accept `bfloat16` inputs?

XLA provides three precision levels that trade MXU passes for mantissa bits:

| Precision Level | TorchTPU Setting | JAX Setting | MXU Passes | Effective Mantissa Precision | Relative Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Default (`bf16`)** | `torch.set_float32_matmul_precision("medium")` | `jax.lax.Precision.DEFAULT` | **1 pass** | ~7 bits (`bfloat16` inputs, `float32` accumulation) | **100% (1.0x)** |
| **High (`bf16_3x`)** | `torch.set_float32_matmul_precision("high")` | `jax.lax.Precision.HIGH` | **3 passes** | ~14 bits ($A_{\text{hi}}B_{\text{hi}} + A_{\text{hi}}B_{\text{lo}} + A_{\text{lo}}B_{\text{hi}}$) | **~33% (0.33x)** |
| **Highest (`bf16_6x`)** | `torch.set_float32_matmul_precision("highest")` or `with torch.tpu.precision(torch.tpu.Precision.HIGHEST):` | `jax.lax.Precision.HIGHEST` | **6 passes** | ~23 bits (Full IEEE `float32` parity) | **~16% (0.16x)** |

### How Multi-Pass Residual Decomposition Works
To multiply two 23-bit mantissa `float32` numbers $a$ and $b$ using a 7-bit `bfloat16` multiplier:
1. XLA splits $a$ into its top 7 bits $a_{\text{hi}} = \text{bf16}(a)$ and residual $a_{\text{lo}} = \text{bf16}(a - a_{\text{hi}})$ (and similarly $b = b_{\text{hi}} + b_{\text{lo}}$).
2. Because $(a_{\text{hi}} + a_{\text{lo}})(b_{\text{hi}} + b_{\text{lo}}) = a_{\text{hi}}b_{\text{hi}} + a_{\text{hi}}b_{\text{lo}} + a_{\text{lo}}b_{\text{hi}} + a_{\text{lo}}b_{\text{lo}}$, executing **3 MXU passes** recovers the top 14+ bits of mantissa precision, and a 3-way split (**6 MXU passes**) recovers full IEEE `float32` accuracy.

### Explicit Precision APIs in Code

#### TorchTPU (`torch_tpu`)
```python
# Global setting for float32 matmuls: "medium" (1-pass), "high" (3-pass), "highest" (6-pass)
torch.set_float32_matmul_precision("high")

# Scoped context manager for fine-grained control over DotGeneral (matmul) and Conv:
if hasattr(torch, "tpu") and hasattr(torch.tpu, "precision"):
    with torch.tpu.precision(torch.tpu.Precision.HIGHEST):
        y_exact = a_fp32 @ b_fp32
```

### Preserving FP32 MXU Accumulators (`out_dtype=torch.float32` & `allow_excess_precision`)
Even when inputs are `bfloat16`, the TPU MXU hardware accumulates inner dot products in **full `float32`** before downcasting the final result back to `bfloat16` at the array boundary. Per `docs/numerical_precision.md`, TorchTPU exposes two key controls over this accumulator behavior:

1. **Direct FP32 Accumulator Readout (`out_dtype=torch.float32`)**:
   In numerically sensitive layers (such as the final `lm_head` vocabulary projection or attention logit computation), casting `bfloat16` inputs to `float32` *before* `torch.mm` triggers slower multi-pass emulation or extra HBM bandwidth. Instead, pass `out_dtype=torch.float32` to `torch.mm` or `torch.bmm` to run a **1-pass `bfloat16` MXU matmul while reading out the native `float32` MXU accumulator directly**:
   ```python
   # Runs at full 1-pass bfloat16 MXU speed (918 TFLOP/s on v6e) and returns exact float32 accumulators!
   logits_fp32 = torch.mm(hidden_bf16, lm_head_weight_bf16.t(), out_dtype=torch.float32)
   ```
2. **Controlling Fused Epilogue Excess Precision (`torch.tpu.allow_excess_precision`)**:
   When XLA fuses elementwise operations (like bias add or residual add) into the epilogue of a `bfloat16` matmul, XLA can either round the matmul output to `bfloat16` immediately (strict PyTorch eager semantics, `torch.tpu.allow_excess_precision = False`, the default) or keep the intermediate value in `float32` across the fused epilogue (`torch.tpu.allow_excess_precision = True`).
   > **Warning — Locked on First Compile:** `torch.tpu.allow_excess_precision` is **locked permanently on the first `torch.compile()` invocation** in a process. Attempting to mutate `torch.tpu.allow_excess_precision` after any graph has been compiled raises a `RuntimeError`. Always configure it at the very top of your entrypoint script.

#### JAX
```python
from jax import lax
y_default = jnp.matmul(a_fp32, b_fp32, precision=lax.Precision.DEFAULT)  # 1-pass
y_high    = jnp.matmul(a_fp32, b_fp32, precision=lax.Precision.HIGH)     # 3-pass
y_highest = jnp.matmul(a_fp32, b_fp32, precision=lax.Precision.HIGHEST)  # 6-pass
```

## 5. Guided Hands-On Lab (`labs/lab_01_device_and_mxu_precision.py`)

Look at `labs/lab_01_device_and_mxu_precision.py` in the right-hand code editor:
1. **Task 1 (Bit-Level Inspection)**: Inspects the raw IEEE binary exponent and mantissa bit patterns of `float32`, `float16`, and `bfloat16`, demonstrating why `1e30` overflows to `inf` in `float16` while surviving intact in `bfloat16`.
2. **Task 2 (Precision vs. Throughput Benchmark)**: Multiplies two matrices in `bfloat16` and `float32` (`DEFAULT` / `medium`, `HIGH` / `high`, and `HIGHEST` / `highest`) in both PyTorch and JAX, measuring numerical error against a `float64` reference alongside achieved **GFLOPs/s (or TFLOPs/s)**.
3. **Task 3 (Tile Alignment Experiment)**: Compares matrix multiplication efficiency on aligned (`512x512`) vs. unaligned (`500x500`) shapes.

Click **Run Code** to execute the benchmark and inspect the precision error vs. throughput table.
