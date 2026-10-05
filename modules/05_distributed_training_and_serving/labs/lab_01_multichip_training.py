#!/usr/bin/env python3
'''Module 5, Lab 1: Multi-Device Sharding, AllReduce (pmean) & GSPMD PartitionSpec.'''

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
try:
  from jax.experimental.shard_map import shard_map
except ImportError:
  shard_map = jax.shard_map


def run_sharding_and_collectives():
  devices = jax.devices()
  num_devs = len(devices)
  print("=" * 78)
  print(f"MULTI-DEVICE SHARDING & COLLECTIVES LAB (device_count={num_devs}, backend={jax.default_backend()})")
  print("=" * 78)
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
    print(f"   Device {shard.device} holds slice {shard.index} -> values={np.asarray(shard.data).squeeze().tolist()}")

  # Explicit collective inside shard_map: each device computes local gradient and runs lax.pmean (AllReduce)
  def _ddp_grad_body(x_local, w_local):
    local_grad = x_local.T @ x_local @ w_local
    # AllReduce mean across 'data' mesh axis
    return jax.lax.pmean(local_grad, axis_name="data")

  distributed_data_parallel_grad = jax.jit(
      shard_map(_ddp_grad_body, mesh=mesh, in_specs=(P("data", None), P()), out_specs=P())
  )

  ar_grad = distributed_data_parallel_grad(x_sharded, w_rep)
  print(f"\n2. AllReduce (lax.pmean across {num_devs} devices) Global Gradient shape={ar_grad.shape}:")
  print(ar_grad)


if __name__ == "__main__":
  run_sharding_and_collectives()
