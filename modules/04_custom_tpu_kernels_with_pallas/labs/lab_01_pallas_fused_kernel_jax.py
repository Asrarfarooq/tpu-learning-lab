#!/usr/bin/env python3
'''Module 4, Lab 1: Writing a Tiled TPU Kernel in JAX Pallas (VMEM Refs & BlockSpec).'''

import numpy as np


def run_pallas_lab():
  try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
  except ImportError:
    print("[JAX / Pallas] Not installed; skipping.")
    return

  backend = jax.default_backend()
  use_interpret = (backend == "cpu")

  print("=" * 78)
  print(f"JAX PALLAS TILED VMEM KERNEL (backend={backend}, interpret={use_interpret})")
  print("=" * 78)

  # Total matrix shape: (512, 512) -> Tiled into (128, 128) VMEM blocks over a (4, 4) grid
  m, n = 512, 512
  bm, bn = 128, 128
  grid = (m // bm, n // bn)

  def fused_gated_silu_kernel(x_ref, gate_ref, o_ref):
    # x_ref and gate_ref live in on-chip VMEM SRAM!
    x_tile = x_ref[:, :]
    g_tile = gate_ref[:, :]
    sig = 1.0 / (1.0 + jnp.exp(-g_tile))
    o_ref[:, :] = (g_tile * sig) * x_tile

  @jax.jit
  def pallas_gated_silu(x: jax.Array, gate: jax.Array) -> jax.Array:
    return pl.pallas_call(
        fused_gated_silu_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=[
            pl.BlockSpec(block_shape=(bm, bn), index_map=lambda i, j: (i, j)),
            pl.BlockSpec(block_shape=(bm, bn), index_map=lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec(block_shape=(bm, bn), index_map=lambda i, j: (i, j)),
        interpret=use_interpret,
    )(x, gate)

  key = jax.random.key(0)
  k1, k2 = jax.random.split(key)
  x_in = jax.random.normal(k1, (m, n), dtype=jnp.float32)
  g_in = jax.random.normal(k2, (m, n), dtype=jnp.float32)

  out_pallas = pallas_gated_silu(x_in, g_in).block_until_ready()
  out_ref = (jax.nn.silu(g_in) * x_in).block_until_ready()

  max_err = float(np.max(np.abs(np.asarray(out_pallas) - np.asarray(out_ref))))
  print(f"  Input HBM Shape : {x_in.shape} ({x_in.nbytes / 1024:.1f} KiB)")
  print(f"  VMEM Tile Shape : ({bm}, {bn}) -> {grid[0]}x{grid[1]} = {grid[0]*grid[1]} tile invocations")
  print(f"  Max Abs Error vs Reference JAX: {max_err:.2e}")


if __name__ == "__main__":
  run_pallas_lab()
