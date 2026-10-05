# Unit 1.1: Systolic Arrays (MXU), `bfloat16` Bit Layout & Multi-Pass Precision

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Explain how the TPU **Matrix Multiply Unit (MXU)** executes matrix multiplications using a 2D **Systolic Array** ($128 \times 128$ on v5e/v5p, $256 \times 256$ on v6e Trillium).
2. Compare the IEEE-754 bit layouts of **`float32`**, **`float16` (FP16)**, and **`bfloat16` (BF16)**, and explain why Google invented `bfloat16` specifically for TPUs.
3. Control TPU matrix multiplication precision in both **TorchTPU** (`torch.set_float32_matmul_precision`, `with torch.tpu.precision(torch.tpu.Precision.HIGHEST):`) and **JAX** (`jax.lax.Precision.DEFAULT`, `HIGH`, `HIGHEST`) and measure the throughput vs. numerical accuracy trade-off.

## 2. Inside the TPU TensorCore: VPU vs. MXU (Systolic Array)

A CPU or GPU core is an instruction-fetching processor: for every fused multiply-add (`FMA`), it fetches instructions, decodes them, reads operands from a register file, writes results back to the register file, and arbitrates cache access.

Google's TPU **Matrix Multiply Unit (MXU)** eliminates register-file read/write overhead during matrix multiplication by using a **2D Systolic Array**:

```
           Weight Matrix W (Pre-loaded / Stationary in MXU)
                 Col 0          Col 1          Col 2  ..127/255
               +--------------+--------------+--------------+
Act Row 0 ---->| W[0,0]       | W[0,1]       | W[0,2]       |
streams right  | Acc += X*W   | Acc += X*W   | Acc += X*W   |
               +------+-------+------+-------+------+-------+
                      |              |              |  Partial sums
                      v              v              v  flow down
Act Row 1 ---->+--------------+--------------+--------------+
streams right  | W[1,0]       | W[1,1]       | W[1,2]       |
               | Acc += X*W   | Acc += X*W   | Acc += X*W   |
               +------+-------+------+-------+------+-------+
                      |              |              |
                      v              v              v
                 Out Y[:,0]     Out Y[:,1]     Out Y[:,2] (fp32)
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

```
IEEE float32 (32 bits = 4 bytes):
+---+----------------------+------------------------------------------+
| S | Exponent (8 bits)    | Mantissa / Fraction (23 bits)            |
| 1 | Range: ~1e-38..3e38  | ~7.2 decimal digits of precision         |
+---+----------------------+------------------------------------------+

IEEE float16 / FP16 (16 bits = 2 bytes):
+---+---------------+------------------------------+
| S | Exp (5 bits)  | Mantissa (10 bits)           |
| 1 | Max: 65,504!  | ~3.3 decimal digits          |
+---+---------------+------------------------------+
  ^-- Gradients underflow/overflow > 65,504 (needs GradScaler)

Google bfloat16 / BF16 (16 bits = 2 bytes):
+---+----------------------+---------------------+
| S | Exponent (8 bits)    | Mantissa (7 bits)   |
| 1 | Range: ~1e-38..3e38  | ~2.4 decimal digits |
+---+----------------------+---------------------+
  ^-- Same 8-bit exponent range as float32! Zero loss-scaling needed.

FP8 E4M3fn / E5M2 (8 bits = 1 byte, supported on v5p & v6e MXU):
+---+-------------+---------------+   +---+-------------+-------------+
| S | Exp (4b)    | Mantissa (3b) |   | S | Exp (5b)    | Mant (2b)   |
| 1 | Max: 448    | Weights / Act |   | 1 | Max: 57,344 | Gradients   |
+---+-------------+---------------+   +---+-------------+-------------+
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
