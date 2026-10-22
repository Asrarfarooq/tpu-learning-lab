#!/usr/bin/env python3
"""Module 1, Lab 3: Inspecting StableHLO / Post-Optimization HLO & Fixing Recompilation Storms (TorchTPU & JAX)."""

import time
import numpy as np


def next_bucket(n: int, buckets=(64, 128, 256)) -> int:
  for b in buckets:
    if n <= b:
      return b
  return buckets[-1]


def run_pytorch_recompilation_and_bucketing():
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
  backend_name = "tpu" if has_tpu else "inductor"

  def sync():
    if has_tpu:
      torch.accelerator.synchronize()

  print("=" * 80)
  print(f"PART 1 (PYTORCH / TORCHTPU): RECOMPILATION STORM vs. BUCKET PADDING ({device})")
  print("=" * 80)

  w = torch.ones((128, 128), dtype=torch.float32, device=device) * 0.05
  b = torch.ones((128,), dtype=torch.float32, device=device) * 0.01

  compile_traces = {"unpadded": 0, "bucketed": 0}

  def raw_block_unpadded(x, weight, bias):
    compile_traces["unpadded"] += 1
    return torch.tanh(x @ weight + bias) * 0.5

  def raw_block_bucketed(x, weight, bias):
    compile_traces["bucketed"] += 1
    return torch.tanh(x @ weight + bias) * 0.5

  compiled_unpadded = torch.compile(raw_block_unpadded, backend=backend_name, dynamic=False)
  compiled_bucketed = torch.compile(raw_block_bucketed, backend=backend_name, dynamic=False)

  seq_lengths = [17, 29, 43, 61, 78, 95, 110, 125]

  # 1A. Anti-Pattern: Variable unpadded sequence lengths under dynamic=False
  t0 = time.perf_counter()
  for seq_len in seq_lengths:
    x_var = torch.ones((seq_len, 128), dtype=torch.float32, device=device)
    _ = compiled_unpadded(x_var, w, b)
    sync()
  unpadded_ms = (time.perf_counter() - t0) * 1e3

  # 1B. Best Practice: Pad sequence length to fixed power-of-2 buckets (64, 128)
  # Warmup the 2 bucket shapes once
  for b_len in (64, 128):
    _ = compiled_bucketed(torch.zeros((b_len, 128), dtype=torch.float32, device=device), w, b)
  sync()
  traces_after_warmup = compile_traces["bucketed"]

  t0 = time.perf_counter()
  for seq_len in seq_lengths:
    b_len = next_bucket(seq_len)
    x_pad = torch.zeros((b_len, 128), dtype=torch.float32, device=device)
    x_pad[:seq_len, :] = 1.0
    out_pad = compiled_bucketed(x_pad, w, b)[:seq_len]
    sync()
  bucketed_ms = (time.perf_counter() - t0) * 1e3

  print(f"  Variable Sequence Lengths Tested : {seq_lengths}")
  print(
      f"  [Anti-Pattern: Unpadded Shapes]  Traces: {compile_traces['unpadded']} compiles |"
      f" Total Time: {unpadded_ms:7.2f} ms"
  )
  print(
      f"  [Best Practice: Bucket Padding]  Traces: {traces_after_warmup} warmup + "
      f"{compile_traces['bucketed'] - traces_after_warmup} steady-state compiles | Total Time: {bucketed_ms:7.2f} ms\n"
  )


def run_jax_hlo_and_recompiles():
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed; skipping JAX section.")
    return

  @jax.jit
  def fused_block(x, w, b):
    return jnp.tanh(x @ w + b) * 0.5

  print("=" * 80)
  print(f"PART 2 (JAX): LOWERED STABLEHLO vs. POST-OPTIMIZATION HLO ({jax.default_backend()})")
  print("=" * 80)

  x0 = jnp.ones((64, 128), dtype=jnp.float32)
  w0 = jnp.ones((128, 128), dtype=jnp.float32) * 0.05
  b0 = jnp.ones((128,), dtype=jnp.float32) * 0.01

  lowered = fused_block.lower(x0, w0, b0)
  stablehlo_str = lowered.as_text()
  compiled = lowered.compile()
  opt_hlo_str = compiled.as_text()

  print("--- Pre-Optimization StableHLO (First 12 lines) ---")
  for line in stablehlo_str.strip().splitlines()[:12]:
    print("  " + line)

  print("\n--- Post-Optimization Fused XLA HLO (First 16 lines) ---")
  for line in opt_hlo_str.strip().splitlines()[:16]:
    print("  " + line)

  print("\n" + "=" * 80)
  print("PART 3 (JAX): DYNAMIC SHAPE RECOMPILATION STORM vs. BUCKET PADDING")
  print("=" * 80)

  seq_lengths = [17, 29, 43, 61, 78, 95, 110, 125]
  compile_count = {"unpadded": 0, "bucketed": 0}

  @jax.jit
  def unpadded_fn(x, w, b):
    compile_count["unpadded"] += 1
    return jnp.tanh(x @ w + b) * 0.5

  @jax.jit
  def bucketed_fn(x, w, b):
    compile_count["bucketed"] += 1
    return jnp.tanh(x @ w + b) * 0.5

  t0 = time.perf_counter()
  for seq_len in seq_lengths:
    x_var = jnp.ones((seq_len, 128), dtype=jnp.float32)
    _ = unpadded_fn(x_var, w0, b0).block_until_ready()
  unpadded_ms = (time.perf_counter() - t0) * 1e3

  for b_len in (64, 128):
    _ = bucketed_fn(jnp.zeros((b_len, 128), dtype=jnp.float32), w0, b0).block_until_ready()
  compiles_after_warmup = compile_count["bucketed"]

  t0 = time.perf_counter()
  for seq_len in seq_lengths:
    b_len = next_bucket(seq_len)
    x_raw = np.ones((seq_len, 128), dtype=np.float32)
    x_pad = np.pad(x_raw, ((0, b_len - seq_len), (0, 0)))
    out_padded = bucketed_fn(jnp.asarray(x_pad), w0, b0).block_until_ready()
    _ = out_padded[:seq_len]
  bucketed_ms = (time.perf_counter() - t0) * 1e3

  print(f"  [Anti-Pattern: Unpadded Shapes]  Recompiles: {compile_count['unpadded']} / {len(seq_lengths)} | Time: {unpadded_ms:7.2f} ms")
  print(f"  [Best Practice: Bucket Padding]  Recompiles during loop: {compile_count['bucketed'] - compiles_after_warmup} | Time: {bucketed_ms:7.2f} ms")
  print(f"  [Speedup from Bucket Padding]    {unpadded_ms / max(bucketed_ms, 1e-6):.2f}x faster!")


if __name__ == "__main__":
  run_pytorch_recompilation_and_bucketing()
  run_jax_hlo_and_recompiles()
