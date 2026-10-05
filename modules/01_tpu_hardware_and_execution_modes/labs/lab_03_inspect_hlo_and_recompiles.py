#!/usr/bin/env python3
'''Module 1, Lab 3: Inspecting XLA StableHLO/HLO Dumps & Dynamic Shape Recompilations.'''

import time


def inspect_jax_hlo_and_recompiles():
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed; skipping.")
    return

  print("=" * 78)
  print("PART 1: INSPECTING LOWERED STABLEHLO & COMPILED XLA HLO")
  print("=" * 78)

  @jax.jit
  def fused_residual_layer(x, w, bias):
    proj = x @ w + bias
    activated = jax.nn.silu(proj)
    return activated + x

  x_sample = jnp.ones((256, 256), dtype=jnp.bfloat16)
  w_sample = jnp.ones((256, 256), dtype=jnp.bfloat16)
  b_sample = jnp.zeros((256,), dtype=jnp.bfloat16)

  lowered = fused_residual_layer.lower(x_sample, w_sample, b_sample)
  stablehlo_str = lowered.as_text()
  print("--- StableHLO (Before XLA Target-Specific Fusion) ---")
  for line in stablehlo_str.strip().splitlines()[:25]:
    print("  " + line)

  compiled = lowered.compile()
  hlo_str = compiled.as_text()
  print("\n--- Post-Optimization XLA HLO (Notice 'dot' + 'fusion' kLoop) ---")
  for line in hlo_str.strip().splitlines()[:35]:
    print("  " + line)

  mem_analysis = compiled.memory_analysis()
  if mem_analysis is not None:
    print("\n--- XLA Static Memory Analysis ---")
    print(f"  Output size in bytes   : {mem_analysis.output_size_in_bytes}")
    print(f"  Temp scratchpad bytes  : {mem_analysis.temp_size_in_bytes}")
    print(f"  Argument size in bytes : {mem_analysis.argument_size_in_bytes}")

  print("\n" + "=" * 78)
  print("PART 2: DYNAMIC SHAPE RECOMPILATION TRAP vs. STATIC BUCKET PADDING")
  print("=" * 78)

  trace_counter = {"count": 0}

  @jax.jit
  def traced_fn(inp, weight):
    trace_counter["count"] += 1
    return jnp.tanh(inp @ weight)

  print("[Trap] Calling @jax.jit with varying batch sizes (31, 32, 33, 34):")
  for b_sz in (31, 32, 33, 34):
    x_var = jnp.ones((b_sz, 256), dtype=jnp.float32)
    t0 = time.perf_counter()
    traced_fn(x_var, jnp.ones((256, 256), dtype=jnp.float32)).block_until_ready()
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  Batch size={b_sz:2d} | Latency={dt:7.2f} ms | Total XLA Compilations Triggered={trace_counter['count']}")

  print("\n[Fix] Padding all batches to static bucket shape (64, 256):")
  for step_idx in range(4):
    x_padded = jnp.ones((64, 256), dtype=jnp.float32)
    t0 = time.perf_counter()
    traced_fn(x_padded, jnp.ones((256, 256), dtype=jnp.float32)).block_until_ready()
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  Step {step_idx} (shape=64x256) | Latency={dt:7.2f} ms | Total XLA Compilations Triggered={trace_counter['count']}")


if __name__ == "__main__":
  inspect_jax_hlo_and_recompiles()
