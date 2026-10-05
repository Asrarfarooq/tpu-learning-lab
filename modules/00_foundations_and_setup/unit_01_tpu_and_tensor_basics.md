# Unit 0.1: Tensors, Memory Layouts & TPU Hardware Hierarchy

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Explain the internal memory representation of a **Tensor** (`torch.Tensor` in PyTorch and `jax.Array` in JAX): contiguous storage buffer, `shape`, `stride`, `dtype`, and `device`.
2. Calculate the exact **High Bandwidth Memory (HBM)** byte footprint of model weights and activations across `float32`, `bfloat16`, and `int8`.
3. Trace how data moves across the physical hardware boundary from **Host x86 CPU DRAM** over **PCIe** into **TPU HBM**, **VMEM (Vector Memory)**, and the **Matrix Multiply Unit (MXU)**.
4. Write and execute side-by-side **TorchTPU (`torch_tpu`)** and **JAX** code for device discovery, tensor allocation, matrix multiplication, and random number generation.

## 2. First-Principles Terminology: What Is a Tensor?

If you come from general software engineering or Python scripting, you are familiar with Python lists (`[1.0, 2.0, 3.0]`). A standard Python list of 1 million floats does not store 1 million raw IEEE-754 floating-point values side-by-side in memory; instead, it stores an array of 1 million 64-bit pointers, each pointing to a separately heap-allocated `PyFloatObject` (24 bytes of reference count, type pointer, and value). Iterating over that list in Python incurs pointer-chasing cache misses and bytecode dispatch on every single element.

A **Tensor** (`torch.Tensor` or `jax.Array`) eliminates this overhead by separating **metadata** from a **raw, contiguous byte buffer**:

```
+-------------------------------------------------------------------+
| Tensor Header (Metadata in Host / Device Runtime)                 |
| - data_ptr : 0x7f8a40000000 (Physical address in RAM or TPU HBM)  |
| - dtype    : bfloat16 (2 bytes per element)                       |
| - shape    : (2, 4)   (2 rows, 4 columns -> 8 elements total)     |
| - strides  : (4, 1)   (Jump 4 elements next row, 1 next col)      |
| - device   : tpu:0    (Resides on TPU Chip 0 HBM)                 |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------+-------+-------+-------+-------+-------+-------+-------+
| [0,0] | [0,1] | [0,2] | [0,3] | [1,0] | [1,1] | [1,2] | [1,3] |
+-------+-------+-------+-------+-------+-------+-------+-------+
Byte 0  Byte 2  Byte 4  Byte 6  Byte 8  Byte 10 Byte 12 Byte 14
```

### Core Tensor Attributes Defined

| Attribute | PyTorch API | JAX API | Engineering Meaning |
| :--- | :--- | :--- | :--- |
| **Rank (`ndim`)** | `x.ndim` | `x.ndim` | Number of dimensions (axes). Scalar = `0`, Vector = `1`, 2D Matrix = `2`, Transformer activation batch `[Batch, Seq, Hidden]` = `3`. |
| **Shape** | `x.shape` | `x.shape` | Tuple of integers giving the length along each axis, e.g., `(32, 2048, 4096)`. |
| **Element Count** | `x.numel()` | `x.size` | Product of all dimensions in `shape`. For `(32, 2048, 4096)`, `numel = 268,435,456`. |
| **Data Type (`dtype`)** | `x.dtype` | `x.dtype` | Binary format of each element: `float32` (4B), `bfloat16` (2B), `float16` (2B), `fp8_e4m3fn` (1B), `fp8_e5m2` (1B), `int8` (1B). |
| **Total Bytes** | `x.numel() * x.element_size()` | `x.nbytes` | Exact memory footprint in bytes: $\text{Element Count} \times \text{Bytes per Element}$. |
| **Device Placement** | `x.device` | `x.devices()` | Which physical memory pool holds the buffer (`cpu` vs. `tpu:0`). |

## 3. Physical Hardware Architecture: CPU DRAM vs. TPU HBM & VMEM

When you SSH into a Cloud TPU VM and run `python3 script.py`, your Python interpreter executes on the **Host x86 CPU** (for example, an AMD EPYC or Intel Xeon processor with 100–400+ GB of system DDR5 RAM). Attached to that host motherboard over **PCIe Gen4/Gen5** are the **Google TPU chips**.

```
+-------------------------------------------------------------------+
| HOST x86 SERVER: System DRAM (100 - 400 GB)                       |
| - Runs Python interpreter, OS, DataLoader & graph dispatch        |
+-------------------------------------------------------------------+
          |  .to("tpu") / jax.device_put()      ^  .cpu() / device_get
          |  PCIe Bus (~16 - 64 GB/s)           |
          v                                     |
+-------------------------------------------------------------------+
| GOOGLE CLOUD TPU CHIP                                             |
|                                                                   |
|  +-------------------------------------------------------------+  |
|  | 1. On-Package HBM (High Bandwidth Memory)                   |  |
|  |    - Capacity  : 32 GB (v6e) / 95 GB (v5p)                  |  |
|  |    - Bandwidth : 1,640 GB/s (v6e) / 2,765 GB/s (v5p)        |  |
|  |    - Stores all TPU tensors (weights, activations, KV cache)|  |
|  +------------------------------+------------------------------+  |
|                                 | Async DMA Engine (Tile Loads)   |
|                                 v                                 |
|  +-------------------------------------------------------------+  |
|  | 2. On-Die VMEM (Vector SRAM Scratchpad)                     |  |
|  |    - Multi-TB/s on-die bandwidth, tens of MB capacity       |  |
|  +------------------------------+------------------------------+  |
|                                 |                                 |
|                                 v                                 |
|  +-------------------------------------------------------------+  |
|  | 3. TensorCore Compute Units                                 |  |
|  |    - MXU : Systolic Matrix Multiply (128x128 v5p / 256x256) |  |
|  |    - VPU : Vector Processing Unit (elementwise / reductions)|  |
|  |    - XLU : Transpose / Permute Unit                         |  |
|  +-------------------------------------------------------------+  |
+-------------------------------------------------------------------+
```

### Why This Memory Hierarchy Matters for Your Code
1. **PCIe is a Narrow Straw (~16–64 GB/s)**: Compared to TPU HBM bandwidth (**1,640 GB/s** on `v6e` and **2,765 GB/s** on `v5p`), copying tensors between Host CPU DRAM and TPU HBM is **50x to 100x slower**.
2. **Rule of Thumb**: Allocate or move model weights and input batches to TPU HBM **once**, run the entire forward/backward/optimizer computation on the TPU, and only copy small scalar summaries (like loss every $N$ steps) back to the CPU.
3. **HBM vs. VMEM**: You never manually place standard PyTorch or JAX tensors into `VMEM` (unless writing custom Pallas kernels in Module 4). When you place a tensor on `"tpu"`, it lives in **HBM**. The **XLA compiler** automatically emits Direct Memory Access (DMA) instructions to slice your HBM tensors into tiles, stream them into **VMEM**, compute on them inside the **MXU/VPU**, and write results back to **HBM**.

## 4. Side-by-Side API Mapping: TorchTPU (`torch_tpu`) vs. JAX

TorchTPU registers the `"tpu"` device backend directly into PyTorch's `torch.accelerator` and `PrivateUse1` dispatcher (`torch.tpu`). If you know `torch.cuda`, you already know 95% of the `torch.tpu` API:

| Task | PyTorch with TorchTPU (`torch` + `torch_tpu`) | JAX (`jax` + `jax.numpy as jnp`) |
| :--- | :--- | :--- |
| **Import Framework** | `import torch; import torch_tpu` | `import jax; import jax.numpy as jnp` |
| **Check TPU Availability** | `torch.tpu.is_available()` or `torch.accelerator.is_available()` | `jax.default_backend() == "tpu"` |
| **Count Logical TPU Chips** | `torch.accelerator.device_count()` | `jax.device_count()` |
| **Get Device Handle** | `device = torch.device("tpu:0")` (or `torch.device("tpu")`) | `device = jax.devices("tpu")[0]` |
| **Query Free / Total HBM** | `free_b, total_b = torch.accelerator.get_memory_info(device)` | `stats = device.memory_stats()` (`bytes_in_use`, `bytes_limit`) |
| **Allocate Directly on TPU** | `x = torch.zeros((1024, 1024), dtype=torch.bfloat16, device=device)` | `x = jnp.zeros((1024, 1024), dtype=jnp.bfloat16)` *(defaults to TPU)* |
| **Copy Host CPU -> TPU** | `x_tpu = x_cpu.to(device)` | `x_tpu = jax.device_put(x_cpu, device)` |
| **Copy TPU -> Host CPU** | `x_cpu = x_tpu.cpu()` | `x_cpu = jax.device_get(x_tpu)` |
| **Wait for TPU Completion** | `torch.accelerator.synchronize()` (or `torch.tpu.synchronize()`) | `x_tpu.block_until_ready()` |
| **Set Random Seed** | `torch.manual_seed(42)` / `torch.tpu.manual_seed_all(42)` | `key = jax.random.key(42)` |

### Crucial Philosophical Difference: Mutability & PRNG State

1. **In-Place Mutation**:
   * **PyTorch (`torch_tpu`)** allows in-place tensor mutations (`x[0, 0] = 5.0` or `x.add_(1.0)`).
   * **JAX** arrays are **strictly immutable**. Attempting `x[0, 0] = 5.0` raises `TypeError: JAX arrays are immutable`. Instead, JAX provides functional indexed updates:

```python
x_new = x.at[0, 0].set(5.0)
```

*Note*: When compiled inside `@jax.jit`, XLA detects when `x` is no longer needed and compiles `.at[...].set(...)` into a zero-copy in-place HBM overwrite.

2. **Random Number Generation (PRNG)**:
   * **PyTorch** maintains a hidden global stateful RNG seed per device (`torch.manual_seed(42)`). Calling `torch.randn((2, 2))` twice in a row produces two different random matrices because the internal counter advances automatically.
   * **JAX** uses an **explicit, stateless counter-based PRNG** (`ThreeFry` / `RBG`). A JAX PRNG `key = jax.random.key(42)` is just a deterministic array of two `uint32` integers. If you pass the exact same `key` to `jax.random.normal(key, (2, 2))` twice, you get the **exact same numbers twice**. To get new random numbers, you deterministically split the key:

```python
key, subkey1, subkey2 = jax.random.split(key, num=3)
mat1 = jax.random.normal(subkey1, (2, 2))
mat2 = jax.random.normal(subkey2, (2, 2))
```

## 5. Matrix Multiplication Shapes (`@` / `matmul`)

Matrix multiplication (`A @ B` or `torch.matmul(A, B)` / `jnp.matmul(A, B)`) is the foundational operation of all neural networks (accounting for >90% of FLOPs in Transformers) and is the exact operation accelerated by the TPU's **Matrix Multiply Unit (MXU)**.

For two 2D matrices $A$ of shape `[M, K]` and $B$ of shape `[K, N]`:
* The **inner dimensions ($K$) must match** (this is the *contracting dimension* over which products are summed: $C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} B_{k,j}$).
* The resulting matrix $C = A @ B$ has shape **`[M, N]`**.
* The total number of floating-point operations (**FLOPs**) required for one 2D matrix multiplication $[M, K] \times [K, N]$ is:
  $$\text{FLOPs}_{\text{matmul}} = 2 \times M \times N \times K$$
  *(Why $2 \times$? Because each of the $M \times N$ output elements requires $K$ multiplications and $K$ additions).*

## 6. Guided Hands-On Lab (`labs/lab_01_tensors_and_devices.py`)

Examine the script loaded in the **Interactive Python Lab Runner** on the right (`labs/lab_01_tensors_and_devices.py`).

### Walkthrough of the Lab Code
* **`task_1_hardware_discovery()`**: Queries both PyTorch (`torch.tpu` / `torch.accelerator`) and JAX (`jax.devices()`) to report the active execution device, chip count, and available HBM memory (`torch.accelerator.get_memory_info()`).
* **`task_2_tensor_anatomy()`**: Allocates a `(1024, 1024)` matrix in both `float32` and `bfloat16`, verifying that `bfloat16` cuts HBM consumption from `4,194,304 bytes (4.0 MiB)` down to `2,097,152 bytes (2.0 MiB)`.
* **`task_3_matrix_multiplication()`**: Multiplies an input batch $X$ of shape `[Batch=4, In=8]` by weight matrix $W$ of shape `[In=8, Out=3]` in both PyTorch and JAX, computes the exact FLOP count ($2 \times 4 \times 3 \times 8 = 192$ FLOPs), and verifies numerical agreement between PyTorch and JAX via max absolute difference.
* **`task_4_mutability_and_rng()`**: Contrasts PyTorch's in-place mutation (`t[0, 1] = 42.0`) against JAX's immutable `.at[0, 1].set(42.0)` and demonstrates why reusing an unsplit JAX PRNG key produces identical tensors.

### Run & Verify
1. Click **Run Code** in the right-hand pane.
2. Verify all 4 tasks execute cleanly and inspect the printed memory sizes and FLOP counts.
3. Complete the **Knowledge Check** below to mark Unit 0.1 complete.
