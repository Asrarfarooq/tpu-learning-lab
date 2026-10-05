# Unit 5.1: Inter-Chip Interconnect (ICI) Collectives — `AllReduce`, `ReduceScatter` & `AllGather`

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Explain how Google Cloud TPUs connect multiple chips directly via dedicated **Inter-Chip Interconnect (ICI)** optical/copper links without going through the host CPU or PCIe switch.
2. Trace the 3 core collective communication primitives executed over ICI:
   * **`AllReduce`** (used by Data Parallelism / DDP)
   * **`ReduceScatter`** + **`AllGather`** (used by FSDP / ZeRO-3 and Tensor Parallelism)
3. Calculate when a model requires **FSDP / Sharding** vs. **DDP** based on per-chip HBM capacity (`32 GB` on `v6e`, `95 GB` on `v5p`).

## 2. The Three Collective Primitives Over TPU ICI

```
1. AllGather (Gather shards from N chips so every chip has the full tensor):
   Chip 0: [A0]  \                 Chip 0: [A0, A1, A2, A3]
   Chip 1: [A1]   == AllGather ==> Chip 1: [A0, A1, A2, A3]
   Chip 2: [A2]   /                Chip 2: [A0, A1, A2, A3]
   Chip 3: [A3]  /                 Chip 3: [A0, A1, A2, A3]

2. ReduceScatter (Sum tensors across chips AND shard the result 1/N per chip):
   Chip 0: [G0_a, G0_b, G0_c, G0_d] \                     Chip 0: [Sum_a]
   Chip 1: [G1_a, G1_b, G1_c, G1_d]  == ReduceScatter ==> Chip 1: [Sum_b]
   Chip 2: [G2_a, G2_b, G2_c, G2_d]  /                    Chip 2: [Sum_c]
   Chip 3: [G3_a, G3_b, G3_c, G3_d] /                     Chip 3: [Sum_d]

3. AllReduce = ReduceScatter followed immediately by AllGather!
```

| Parallelism Strategy | What Is Replicated vs. Sharded Across $N$ TPU Chips | Collective Used | HBM Memory per Chip (for $P$ parameters) |
| :--- | :--- | :--- | :--- |
| **DDP (Distributed Data Parallel)** | **Replicated**: Full weights, full grads, full AdamW state on every chip. **Sharded**: Input batch `[B/N, Seq]`. | `AllReduce` on gradients during backward pass. | $16 \times P$ bytes (Cannot fit >2B model on 1 `v6e` 32GB chip). |
| **FSDP / ZeRO-3 (Fully Sharded Data Parallel)** | **Sharded ($1/N$)**: Weights, grads, and AdamW state are all partitioned $1/N$ across chips. | `AllGather` weights before each layer; `ReduceScatter` gradients during backward. | $\frac{16 \times P}{N}$ bytes (Scales linearly with chip count $N$!). |

## 3. Guided Hands-On Lab (`labs/lab_01_multichip_training.py`)

Click **Run Code** to execute `labs/lab_01_multichip_training.py`. It constructs a device mesh across all available devices (`jax.devices()`, or simulates a 4-device mesh when running on a single chip/CPU) and demonstrates `NamedSharding`, `PartitionSpec`, and `shard_map` with `jax.lax.pmean` (`AllReduce`).
