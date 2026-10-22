#!/usr/bin/env python3
"""Module 5, Lab 1: Multi-Device Sharding, AllGather, ReduceScatter & AllReduce Identity (TorchTPU & JAX)."""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np


def run_pytorch_collectives_and_topology():
  try:
    import torch
  except ImportError:
    print("[PyTorch] Not installed; skipping PyTorch section.\n")
    return

  try:
    import torch_tpu  # noqa: F401
  except ImportError:
    pass

  has_tpu = (
      hasattr(torch, "accelerator")
      and torch.accelerator.is_available()
      and getattr(torch.accelerator.current_accelerator(), "type", None) == "tpu"
  )
  device = torch.device("tpu:0" if has_tpu else "cpu")

  print("=" * 88)
  print(f"PART 1 (PYTORCH / TORCHTPU): RING COLLECTIVE IDENTITY & TOPOLOGY ({device})")
  print("=" * 88)

  if has_tpu and hasattr(torch, "tpu") and hasattr(torch.tpu, "distributed"):
    chip_idx = torch.tpu.distributed.physical_chip_index()
    chip_coords = torch.tpu.distributed.chip_coordinates()
    print(f"  Physical Chip Index: {chip_idx} | Torus Coordinates: {chip_coords}")

  # Simulate 4-rank sharded gradient tensors [world_size=4, rows=4, cols=4] in PyTorch
  world_size = 4
  torch.manual_seed(0)
  per_rank_grads = [torch.arange(r * 16, (r + 1) * 16, dtype=torch.float32, device=device).reshape(4, 4) for r in range(world_size)]

  # 1. Direct AllReduce (sum across all 4 ranks)
  all_reduced = torch.stack(per_rank_grads, dim=0).sum(dim=0)

  # 2. ReduceScatter (each rank r receives sum of row slice [r:r+1, :]) followed by AllGather
  reduce_scattered_shards = [
      torch.stack([g[r : r + 1, :] for g in per_rank_grads], dim=0).sum(dim=0)
      for r in range(world_size)
  ]
  all_gathered = torch.cat(reduce_scattered_shards, dim=0)

  pt_diff = float((all_reduced - all_gathered).abs().max().cpu().item())
  print(f"  Simulated World Size                  : {world_size} ranks")
  print(f"  ReduceScatter Shard Shape per Rank    : {tuple(reduce_scattered_shards[0].shape)} (1/{world_size} of [4, 4])")
  print(f"  AllGather(ReduceScatter) vs AllReduce : Max Abs Diff = {pt_diff:.2e} (Exact identity!)\n")


def run_jax_sharding_and_collectives():
  try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    try:
      from jax.experimental.shard_map import shard_map
    except ImportError:
      shard_map = jax.shard_map
  except ImportError:
    print("[JAX] Not installed; skipping JAX section.")
    return

  devices = jax.devices()
  num_devs = len(devices)
  print("=" * 88)
  print(
      f"PART 2 (JAX): MULTI-DEVICE SHARDING & ICI COLLECTIVES (device_count={num_devs},"
      f" backend={jax.default_backend()})"
  )
  print("=" * 88)
  print(f"Logical Devices: {[str(d) for d in devices]}")

  mesh = Mesh(np.array(devices), ("data",))
  data_sharding = NamedSharding(mesh, P("data", None))
  replicated_sharding = NamedSharding(mesh, P())

  batch_x = jnp.arange(num_devs * 4, dtype=jnp.float32).reshape(num_devs, 4)
  w = jnp.ones((4, 4), dtype=jnp.float32)

  x_sharded = jax.device_put(batch_x, data_sharding)
  w_rep = jax.device_put(w, replicated_sharding)

  print(f"\n1. Sharded Input Batch X shape={x_sharded.shape}, sharding={x_sharded.sharding}")
  for shard in x_sharded.addressable_shards:
    print(
        f"   Device {shard.device} holds slice {shard.index} ->"
        f" values={np.asarray(shard.data).squeeze().tolist()}"
    )

  def _allreduce_body(x_local, w_local):
    local_grad = x_local.T @ x_local @ w_local
    return jax.lax.psum(local_grad, axis_name="data")

  def _rs_then_ag_body(x_local, w_local):
    local_grad = x_local.T @ x_local @ w_local
    scattered = jax.lax.psum_scatter(
        local_grad, axis_name="data", scatter_dimension=0, tiled=True
    )
    gathered = jax.lax.all_gather(
        scattered, axis_name="data", axis=0, tiled=True
    )
    return gathered

  from jax.experimental.shard_map import shard_map
  ar_fn = jax.jit(
      shard_map(_allreduce_body, mesh=mesh, in_specs=(P("data", None), P()), out_specs=P(), check_rep=False)
  )
  rs_ag_fn = jax.jit(
      shard_map(_rs_then_ag_body, mesh=mesh, in_specs=(P("data", None), P()), out_specs=P(), check_rep=False)
  )

  ar_grad = ar_fn(x_sharded, w_rep)
  rs_ag_grad = rs_ag_fn(x_sharded, w_rep)
  max_diff = float(jnp.max(jnp.abs(ar_grad - rs_ag_grad)))

  print(f"\n2. AllReduce(G) vs. AllGather(ReduceScatter(G)) across {num_devs} devices:")
  print(f"   AllReduce Output [0]:                 {np.asarray(ar_grad[0]).tolist()}")
  print(f"   AllGather(ReduceScatter) Output [0]:  {np.asarray(rs_ag_grad[0]).tolist()}")
  print(f"   Max Absolute Difference:              {max_diff:.2e} (Exact mathematical identity verified!)")


if __name__ == "__main__":
  run_pytorch_collectives_and_topology()
  run_jax_sharding_and_collectives()
