# Unit 5.1: Inter-Chip Interconnect (ICI) Collectives — `AllReduce`, `ReduceScatter` & `AllGather`

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Explain how Google Cloud TPUs connect multiple chips directly via dedicated **Inter-Chip Interconnect (ICI)** optical/copper links (`2D Torus` on `v6e`, `3D Torus` on `v5p`) without traversing the host CPU or PCIe switch.
2. Apply the **$\alpha + \beta B$ Ring Collective Cost Model** to calculate exact communication time for **`AllGather`**, **`ReduceScatter`**, **`AllReduce`**, and **`AllToAll`**.
3. Derive the **Compute-to-Communication Overlap Condition** for FSDP (`AllGather` hidden behind MXU Matmul) and calculate exact per-chip HBM memory footprints across **DDP** vs. **FSDP / ZeRO-3**.

## 2. The Three Core Collective Primitives Over TPU ICI

```schematic
ici-collectives
```

### The $\alpha + \beta B$ Ring Collective Cost Model (JAX Scaling Book)
Suppose $N$ TPU chips are connected in a 1D ring along an ICI axis where each link has per-hop startup latency $\alpha$ ($\sim 1\text{ }\mu\text{s}$) and unidirectional link bandwidth $\beta_{\text{ICI}}$ (GB/s), and the total tensor size is $B$ bytes ($B / N$ bytes per shard):

1. **`ReduceScatter` or `AllGather` Time ($N - 1$ ring steps of shard size $B/N$)**:
   $$T_{\text{AG}}(B, N) = T_{\text{RS}}(B, N) = (N - 1)\alpha + \frac{N - 1}{N} \frac{B}{\beta_{\text{ICI}}} \approx \frac{B}{\beta_{\text{ICI}}} \quad (\text{for large } B)$$
2. **`AllReduce` Time (`ReduceScatter` followed by `AllGather`)**:
   $$T_{\text{AR}}(B, N) = 2(N - 1)\alpha + 2 \frac{N - 1}{N} \frac{B}{\beta_{\text{ICI}}} \approx \frac{2B}{\beta_{\text{ICI}}}$$
3. **`AllToAll` Time (Used in Mixture-of-Experts token routing & Sequence Parallelism)**:
   Every chip sends a distinct $B / N^2$ slice to every other chip. On a ring of $N$ chips, average hop distance is $N/4$ (bidirectional), giving $T_{\text{A2A}} \approx \frac{B}{4 \beta_{\text{ICI}}}$.

## 3. DDP vs. FSDP Memory & Communication Roofline Math

For a model with $P$ parameters trained in mixed precision (`bfloat16` weights + `bfloat16` gradients + `float32` master weights and AdamW 1st/2nd moments $m, v = 4 + 4 + 4 = 12\text{ B}$, or `bfloat16` moments via `TorchAdamw(use_bfloat16_moments=True)`):

| Parallelism Strategy | What Is Replicated vs. Sharded Across $N$ TPU Chips | Collective Executed per Layer | Static HBM Memory per Chip (Standard FP32 AdamW = $16P$ B) | Static HBM per Chip (`use_bfloat16_moments` = $8P$ B) |
| :--- | :--- | :--- | :--- | :--- |
| **DDP (Distributed Data Parallel)** | **Replicated**: Full weights, grads, and optimizer state on every chip. **Sharded**: Batch `[B/N, S]`. | `AllReduce` on gradients during backward ($2B_{\text{grad}} / \beta_{\text{ICI}}$). | $16 \times P$ bytes *(7B model = **112 GB** $\rightarrow$ OOMs both `v6e` 32GB and `v5p` 95GB!)*. | $8 \times P$ bytes *(7B model = **56 GB** $\rightarrow$ OOMs `v6e` 32GB)*. |
| **FSDP / ZeRO-3 (`SimpleFSDP` / `NamedSharding`)** | **Sharded ($1/N$)**: Weights, grads, and optimizer state are all partitioned $1/N$ across chips. | `AllGather` weights before fwd & bwd ($2 \times B_W / \beta_{\text{ICI}}$) + `ReduceScatter` grads in bwd ($B_W / \beta_{\text{ICI}}$). | $\frac{16 \times P}{N}$ bytes *(7B model on `v6e-8` = **14.0 GB/chip** — fits easily!)*. | $\frac{8 \times P}{N}$ bytes *(7B model on `v6e-8` = **7.0 GB/chip**!)*. |

### Worked Derivation: When Does MXU Compute Hide FSDP `AllGather` Over ICI?
In FSDP, while the MXU is computing Layer $\ell$ (`FLOPs` $= 2 \cdot T_{\text{local}} \cdot D \cdot F$ where $T_{\text{local}} = B_{\text{local}} \cdot S$ tokens per chip), the ICI engine prefetches the `bfloat16` weights of Layer $\ell+1$ (`Bytes` $= 2 \cdot D \cdot F$) via `AllGather`. Communication is **completely hidden behind compute** whenever $T_{\text{compute}} \ge T_{\text{AllGather}}$:

$$\frac{2 \cdot T_{\text{local}} \cdot D \cdot F}{\pi_{\text{MXU}}} \ge \frac{2 \cdot D \cdot F}{\beta_{\text{ICI}}} \iff \mathbf{T_{\text{local}} \ge \frac{\pi_{\text{MXU}}}{\beta_{\text{ICI}}}}$$

* **Engineering Takeaway**: Notice that $D$ and $F$ cancel out! As long as each chip processes at least $T_{\text{local}} \ge \pi_{\text{MXU}} / \beta_{\text{ICI}}$ tokens per step (typically **$T_{\text{local}} \ge 2048\text{–}4096$ tokens per chip**, e.g., `local_batch=2, seq_len=2048`), FSDP's `AllGather` and `ReduceScatter` overlap 100% with MXU matmuls!

## 4. TorchTPU's 3 Distributed Collective Modes (`docs/distributed_support.md`)

In `torch_tpu 0.2.0+`, you can execute collectives over the `"tpu_dist"` PJRT process group using three distinct compilation strategies (`docs/distributed_support.md`):

| Collective Mode | How It Works | When to Use |
| :--- | :--- | :--- |
| **1. Skinny Collectives (Default)** | Standard `torch.distributed` calls (`dist.all_reduce`, `DDP`, `FSDPv2`) cause a graph break at the collective boundary, launch a tiny standalone single-op collective program over PJRT `"tpu_dist"`, and resume compiled compute. | Default safe path for unmodified PyTorch `DDP` or imperative scripts where collectives may only execute on conditional branches. |
| **2. Full-Graph SPMD (`@torch.tpu.distributed.spmd_safe` / `SimpleFSDP` / `DTensor`)** | Functional collectives (`torch.ops._c10d_functional.all_reduce`, `all_gather_into_tensor`, `reduce_scatter_tensor`) and `DTensor.redistribute` are lowered **directly inside the compiled XLA HLO graph** (`all-reduce`, `all-gather`, `reduce-scatter`). | **Maximum performance** (`3.3x` faster on `v6e-8` in `SimpleFSDP`): XLA overlaps ICI DMA transfers with MXU compute inside a single unbroken executable! |
| **3. SPMD Handshake Validation (`TORCH_TPU_SPMD_HANDSHAKE_STAGE`)** | Verifies that every rank in the process group compiles and/or dispatches the exact same XLA collective program ID before launching on TPU hardware (preventing deadlocks if one rank takes a different Python `if` branch). | `COMPILE_STAGE` (default: validates during compilation), `DISPATCH_STAGE` (validates on every launch for debugging rank divergence), or `OFF`. |

### Physical Chip Topology Introspection (`physical_chip_index` & `chip_coordinates`)
When debugging multi-chip ICI ring ordering or pinning custom data loaders per chip, use TorchTPU's public topology introspection helpers (`torch_tpu/tpu/distributed/`):

```python
import torch

chip_idx = torch.tpu.distributed.physical_chip_index()  # e.g. 0..7 on v6e-8
coords   = torch.tpu.distributed.chip_coordinates()     # e.g. (x, y, z) torus coordinates
```

## 5. Guided Hands-On Lab (`labs/lab_01_multichip_training.py`)

Click **Run Code** to execute `labs/lab_01_multichip_training.py`. It demonstrates **both TorchTPU (`tpu_dist` / functional `_c10d_functional` `all_gather_into_tensor` & `reduce_scatter_tensor`) and JAX (`Mesh`, `NamedSharding`, `shard_map`)** side-by-side, and verifies mathematically that `AllGather(ReduceScatter(G)) == AllReduce(G)`.
