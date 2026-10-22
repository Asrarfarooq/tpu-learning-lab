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

```schematic
tensor-memory-layout
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

```schematic
tpu-hardware-hierarchy
```

### Why This Memory Hierarchy Matters for Your Code
1. **PCIe is a Narrow Straw (~16–64 GB/s)**: Compared to TPU HBM bandwidth (**1,640 GB/s** on `v6e` and **2,765 GB/s** on `v5p`), copying tensors between Host CPU DRAM and TPU HBM is **50x to 100x slower**.
2. **Rule of Thumb**: Allocate or move model weights and input batches to TPU HBM **once**, run the entire forward/backward/optimizer computation on the TPU, and only copy small scalar summaries (like loss every $N$ steps) back to the CPU.
3. **HBM vs. VMEM**: You never manually place standard PyTorch or JAX tensors into `VMEM` (unless writing custom Pallas kernels in Module 4). When you place a tensor on `"tpu"`, it lives in **HBM**. The **XLA compiler** automatically emits Direct Memory Access (DMA) instructions to slice your HBM tensors into tiles, stream them into **VMEM**, compute on them inside the **MXU/VPU**, and write results back to **HBM**.

## 4. Side-by-Side API Mapping: TorchTPU (`torch_tpu`) vs. JAX

In the open-source `torch_tpu` (`0.2.0+`) release, installing `torch_tpu` registers PyTorch's `TORCH_DEVICE_BACKEND_AUTOLOAD=1` entry point (`torch_tpu._autoload`). Simply running `import torch` automatically loads the `"tpu"` device backend into PyTorch's unified **`torch.accelerator`** API and configures `torch.compile` to automatically select `backend="tpu"` (`dynamic=False`) whenever inputs reside on a TPU device.

> **Important Public API Boundary (`torch.accelerator` vs. `torch.tpu`):** Beginning with PyTorch 2.11 and `torch_tpu 0.2.0`, `torch.tpu.is_available()`, `torch.tpu.device_count()`, `torch.tpu.current_device()`, and `torch.tpu.synchronize()` are marked **Internal-Only** (`strict=True` in `torch_tpu._internal.public_api`) so that PyTorch's device-agnostic `torch.accelerator` module can delegate to them internally without user code coupling to legacy device-specific calls. Always call **`torch.accelerator.is_available()`**, **`torch.accelerator.device_count()`**, and **`torch.accelerator.synchronize()`** in user code, and use **`torch.tpu.get_device_properties()`** for TPU-specific silicon introspection.

| Task | PyTorch with TorchTPU (`torch` + `torch_tpu`) | JAX (`jax` + `jax.numpy as jnp`) |
| :--- | :--- | :--- |
| **Import Framework** | `import torch` *(autoloads `torch_tpu` via `TORCH_DEVICE_BACKEND_AUTOLOAD=1`)* | `import jax; import jax.numpy as jnp` |
| **Check TPU Availability** | `torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"` | `jax.default_backend() == "tpu"` |
| **Count Logical TPU Chips** | `torch.accelerator.device_count()` | `jax.device_count()` |
| **Get Device Handle** | `device = torch.device("tpu:0")` (or `torch.device("tpu")`) | `device = jax.devices("tpu")[0]` |
| **Hardware Chip Properties** | `props = torch.tpu.get_device_properties(0)` (`.name`, `.hbm_bytes_per_chip`, `.bf16_flops_per_chip`, `.topology()`) | `device.device_kind`, `device.coords` |
| **Query Free / Total HBM** | `free_b, total_b = torch.accelerator.get_memory_info(device)` | `stats = device.memory_stats()` (`bytes_in_use`, `bytes_limit`) |
| **Allocate Directly on TPU** | `x = torch.zeros((1024, 1024), dtype=torch.bfloat16, device=device)` | `x = jnp.zeros((1024, 1024), dtype=jnp.bfloat16)` *(defaults to TPU)* |
| **Copy Host CPU -> TPU** | `x_tpu = x_cpu.to(device)` | `x_tpu = jax.device_put(x_cpu, device)` |
| **Copy TPU -> Host CPU** | `x_cpu = x_tpu.cpu()` | `x_cpu = jax.device_get(x_tpu)` |
| **Wait for TPU Completion** | `torch.accelerator.synchronize()` | `x_tpu.block_until_ready()` |
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
