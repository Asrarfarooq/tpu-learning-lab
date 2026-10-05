# Unit 4.1: Pallas Programming Model — VMEM Refs, `BlockSpec` & Grid Pipelining

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Explain when and why to write a custom **Pallas** kernel (`jax.experimental.pallas`) instead of relying solely on default XLA compiler fusion.
2. Design a **Pallas Grid** and **`pl.BlockSpec`** that tiles large HBM tensors into on-chip **VMEM** scratchpad memory (respecting TPU $(8, 128)$ vector lane alignment).
3. Write and execute a custom Pallas kernel with `pl.pallas_call`.

## 2. How Pallas Works on TPU Hardware

While XLA's automatic `kLoop` fusion handles standard elementwise chains well, complex algorithms like **FlashAttention / SplashAttention**, **Register-Level `int8` Quantization & Bit-Packing**, or **Ragged Paged Attention** require explicit control over **which tiles stay resident in on-chip VMEM and vector registers** so that intermediate buffers are never materialized in HBM.

**Pallas** (`from jax.experimental import pallas as pl`) exposes the TPU's VMEM and DMA engine directly in Python:

```
HBM Tensor X (Shape: [1024, 1024], Grid: 4x4 tiles of 256x256)
+---------+---------+---------+---------+
| (0, 0)  | (0, 1)  | (0, 2)  | (0, 3)  |
+---------+---------+---------+---------+
| (1, 0)  | (1, 1)  | (1, 2)  | (1, 3)  |
+---------+---------+---------+---------+
          |  DMA In (BlockSpec: block_shape=(256, 256))
          v
+---------------------------------------+
| On-Chip VMEM Tile: x_ref[256, 256]    |
| - Computed in VPU / MXU at TB/s speed |
| - Zero intermediate HBM writes        |
+---------------------------------------+
          |  DMA Out
          v
HBM Output Tensor O (Shape: [1024, 1024])
```

### The Three Building Blocks of `pl.pallas_call`
1. **The Kernel Body (`def kernel(x_ref, y_ref, o_ref):`)**:
   * Receives `Ref` objects pointing to **VMEM buffers** (not HBM!).
   * Reads values into vector registers via `x_tile = x_ref[:, :]`, computes on them, and writes results to `o_ref[:, :] = ...`.
2. **The `grid` Tuple**:
   * Specifies how many tile iterations to run, e.g., `grid = (M // bm, N // bn)`.
3. **The `BlockSpec(block_shape=..., index_map=...)`**:
   * Always pass `block_shape` and `index_map` as **keyword arguments** (`pl.BlockSpec(block_shape=(bm, bn), index_map=lambda i, j: (i, j))`) so your code is compatible across all JAX versions.
   * Tells the TPU DMA hardware which slice of the HBM tensor corresponds to grid coordinate `(i, j)`.
   * **TPU Hardware Rule**: On TPU, the last two dimensions of every `block_shape` must be divisible by **`(8, 128)`** (the native shape of a TPU VPU vector register) or equal to the full axis dimension.

## 3. Guided Hands-On Lab (`labs/lab_01_pallas_fused_kernel_jax.py`)

Click **Run Code** to execute `labs/lab_01_pallas_fused_kernel_jax.py`, which writes a tiled Pallas kernel (`fused_gated_silu_kernel`), runs it via `pl.pallas_call` (using `interpret=True` automatically when on CPU and native TPU Mosaic lowering when on a Cloud TPU VM), and verifies zero numerical drift against standard JAX.
