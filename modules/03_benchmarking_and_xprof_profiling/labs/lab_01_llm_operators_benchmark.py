#!/usr/bin/env python3
"""Module 3, Lab 1: Roofline Arithmetic Intensity & LLM Operator Benchmark (TorchTPU & JAX)."""

import time
import numpy as np


def compute_roofline_metrics(name: str, flops: int, bytes_io: int, elapsed_sec: float):
  ai = flops / max(bytes_io, 1)
  gflops_s = (flops / max(elapsed_sec, 1e-9)) / 1e9
  gb_s = (bytes_io / max(elapsed_sec, 1e-9)) / 1e9
  bound = "COMPUTE-BOUND (MXU)" if ai >= 240.0 else "MEMORY-BOUND (HBM)"
  print(f"  {name:<28} | Latency: {elapsed_sec*1e3:7.3f} ms | AI: {ai:6.1f} FLOP/B | {gflops_s:7.1f} GFLOPs/s | {gb_s:6.1f} GB/s | {bound}")


def run_benchmark():
  print("=" * 98)
  print("ROOFLINE & LLM OPERATOR BENCHMARK (PREFILL MATMUL vs. DECODE MATMUL vs. RMSNORM vs. SWIGLU)")
  print("=" * 98)

  k_dim, n_dim = 1024, 1024
  m_prefill = 1024
  m_decode = 1
  iters = 20

  try:
    import jax
    import jax.numpy as jnp

    print(f"[JAX Backend: {jax.default_backend()}]")
    key = jax.random.key(0)
    x_prefill = jax.random.normal(key, (m_prefill, k_dim), dtype=jnp.bfloat16)
    x_decode = jax.random.normal(key, (m_decode, k_dim), dtype=jnp.bfloat16)
    w = jax.random.normal(key, (k_dim, n_dim), dtype=jnp.bfloat16)

    matmul_fn = jax.jit(lambda a, b: a @ b)
    rmsnorm_fn = jax.jit(lambda a: a * jax.lax.rsqrt(jnp.mean(jnp.square(a.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6).astype(a.dtype))
    swiglu_fn = jax.jit(lambda a, w1, w2: jax.nn.silu(a @ w1) * (a @ w2))

    # Warmup
    matmul_fn(x_prefill, w).block_until_ready()
    matmul_fn(x_decode, w).block_until_ready()
    rmsnorm_fn(x_prefill).block_until_ready()
    swiglu_fn(x_prefill, w, w).block_until_ready()

    # 1. Prefill Matmul (M=1024, K=1024, N=1024)
    t0 = time.perf_counter()
    for _ in range(iters):
      matmul_fn(x_prefill, w).block_until_ready()
    dt_pre = (time.perf_counter() - t0) / iters
    flops_pre = 2 * m_prefill * k_dim * n_dim
    bytes_pre = 2 * (m_prefill * k_dim + k_dim * n_dim + m_prefill * n_dim)
    compute_roofline_metrics("1. Prefill Matmul (M=1024)", flops_pre, bytes_pre, dt_pre)

    # 2. Decode Matmul (M=1, K=1024, N=1024)
    t0 = time.perf_counter()
    for _ in range(iters):
      matmul_fn(x_decode, w).block_until_ready()
    dt_dec = (time.perf_counter() - t0) / iters
    flops_dec = 2 * m_decode * k_dim * n_dim
    bytes_dec = 2 * (m_decode * k_dim + k_dim * n_dim + m_decode * n_dim)
    compute_roofline_metrics("2. Decode Matmul  (M=1)", flops_dec, bytes_dec, dt_dec)

    # 3. RMSNorm (M=1024, K=1024)
    t0 = time.perf_counter()
    for _ in range(iters):
      rmsnorm_fn(x_prefill).block_until_ready()
    dt_rms = (time.perf_counter() - t0) / iters
    flops_rms = 4 * m_prefill * k_dim
    bytes_rms = 2 * 2 * m_prefill * k_dim
    compute_roofline_metrics("3. Fused RMSNorm  (M=1024)", flops_rms, bytes_rms, dt_rms)

    # 4. SwiGLU Gate+Up (M=1024, K=1024, N=1024)
    t0 = time.perf_counter()
    for _ in range(iters):
      swiglu_fn(x_prefill, w, w).block_until_ready()
    dt_swi = (time.perf_counter() - t0) / iters
    flops_swi = 4 * m_prefill * k_dim * n_dim + 4 * m_prefill * n_dim
    bytes_swi = 2 * (m_prefill * k_dim + 2 * k_dim * n_dim + m_prefill * n_dim)
    compute_roofline_metrics("4. Fused SwiGLU   (M=1024)", flops_swi, bytes_swi, dt_swi)
  except ImportError:
    print("[JAX] Not installed.")


if __name__ == "__main__":
  run_benchmark()
