#!/usr/bin/env python3
'''Module 1, Lab 2: Eager vs. Compiled Execution Benchmark (TorchTPU & JAX).'''

import time
import numpy as np

BATCH = 512
HIDDEN = 512
ITERS = 25


def benchmark_pytorch():
  try:
    import torch
    import torch.nn.functional as F
  except ImportError:
    print("[PyTorch] Not installed; skipping.")
    return

  has_tpu = False
  try:
    import torch_tpu  # noqa: F401
    has_tpu = hasattr(torch, "tpu") and torch.tpu.is_available()
  except ImportError:
    pass

  device = torch.device("tpu:0" if has_tpu else "cpu")
  backend_name = "tpu" if has_tpu else "inductor"

  def sync():
    if has_tpu:
      torch.tpu.synchronize()

  print("=" * 78)
  print(f"PYTORCH EXECUTION BENCHMARK (device={device}, compile_backend='{backend_name}')")
  print("=" * 78)

  torch.manual_seed(0)
  x = torch.randn((BATCH, HIDDEN), dtype=torch.float32, device=device)
  w1 = torch.randn((HIDDEN, HIDDEN), dtype=torch.float32, device=device) * 0.02
  w2 = torch.randn((HIDDEN, HIDDEN), dtype=torch.float32, device=device) * 0.02

  def neural_block(inp, weight1, weight2):
    h = inp @ weight1
    h = F.gelu(h)
    h = (h * 0.5 + inp) @ weight2
    h = torch.tanh(h) + inp
    mean = h.mean(dim=-1, keepdim=True)
    var = h.var(dim=-1, keepdim=True, unbiased=False)
    return (h - mean) / torch.sqrt(var + 1e-5)

  # 1. Warmup and time Eager Mode
  out_eager = neural_block(x, w1, w2)
  sync()
  t0 = time.perf_counter()
  for _ in range(ITERS):
    out_eager = neural_block(x, w1, w2)
    sync()
  eager_ms = ((time.perf_counter() - t0) / ITERS) * 1e3

  # 2. Cold Start (Step 1) + Warm Cache Hits (Steps 2+) for torch.compile
  compiled_block = torch.compile(neural_block, backend=backend_name, dynamic=False)
  t_cold_start = time.perf_counter()
  out_compiled = compiled_block(x, w1, w2)
  sync()
  cold_compile_ms = (time.perf_counter() - t_cold_start) * 1e3

  t0 = time.perf_counter()
  for _ in range(ITERS):
    out_compiled = compiled_block(x, w1, w2)
    sync()
  compiled_ms = ((time.perf_counter() - t0) / ITERS) * 1e3

  max_diff = float((out_eager - out_compiled).abs().max().cpu().item())
  print(f"  [Eager Mode]          Avg Latency : {eager_ms:8.3f} ms / step")
  print(f"  [torch.compile Cold]  Step 1 Time : {cold_compile_ms:8.3f} ms (Graph capture + compilation)")
  print(f"  [torch.compile Warm]  Avg Latency : {compiled_ms:8.3f} ms / step")
  print(f"  [Speedup & Parity]    Warm Speedup: {eager_ms / max(compiled_ms, 1e-6):.2f}x | Max Diff: {max_diff:.2e}\n")


def benchmark_jax():
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed; skipping.")
    return

  print("=" * 78)
  print(f"JAX EXECUTION BENCHMARK (backend={jax.default_backend()})")
  print("=" * 78)

  key = jax.random.key(0)
  k1, k2, k3 = jax.random.split(key, 3)
  x = jax.random.normal(k1, (BATCH, HIDDEN), dtype=jnp.float32)
  w1 = jax.random.normal(k2, (HIDDEN, HIDDEN), dtype=jnp.float32) * 0.02
  w2 = jax.random.normal(k3, (HIDDEN, HIDDEN), dtype=jnp.float32) * 0.02

  def jax_neural_block(inp, weight1, weight2):
    h = inp @ weight1
    h = jax.nn.gelu(h)
    h = (h * 0.5 + inp) @ weight2
    h = jnp.tanh(h) + inp
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.var(h, axis=-1, keepdims=True)
    return (h - mean) / jnp.sqrt(var + 1e-5)

  # 1. Eager JAX (op-by-op dispatch)
  out_eager = jax_neural_block(x, w1, w2).block_until_ready()
  t0 = time.perf_counter()
  for _ in range(ITERS):
    out_eager = jax_neural_block(x, w1, w2).block_until_ready()
  eager_ms = ((time.perf_counter() - t0) / ITERS) * 1e3

  # 2. @jax.jit Cold Start + Warm Cache Hits
  jit_block = jax.jit(jax_neural_block)
  t_cold = time.perf_counter()
  out_jit = jit_block(x, w1, w2).block_until_ready()
  cold_ms = (time.perf_counter() - t_cold) * 1e3

  t0 = time.perf_counter()
  for _ in range(ITERS):
    out_jit = jit_block(x, w1, w2).block_until_ready()
  jit_ms = ((time.perf_counter() - t0) / ITERS) * 1e3

  max_diff = float(np.max(np.abs(np.asarray(out_eager) - np.asarray(out_jit))))
  print(f"  [JAX Eager Op-by-Op]  Avg Latency : {eager_ms:8.3f} ms / step")
  print(f"  [@jax.jit Cold]       Step 1 Time : {cold_ms:8.3f} ms (XLA compilation)")
  print(f"  [@jax.jit Warm]       Avg Latency : {jit_ms:8.3f} ms / step")
  print(f"  [Speedup & Parity]    Warm Speedup: {eager_ms / max(jit_ms, 1e-6):.2f}x | Max Diff: {max_diff:.2e}\n")


if __name__ == "__main__":
  benchmark_pytorch()
  benchmark_jax()
