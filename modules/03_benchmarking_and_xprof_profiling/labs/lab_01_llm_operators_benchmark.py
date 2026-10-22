#!/usr/bin/env python3
"""Module 3, Lab 1: Roofline Arithmetic Intensity & LLM Operator Benchmark (TorchTPU & JAX)."""

import time
import numpy as np


def compute_roofline_metrics(
    name: str,
    flops: int,
    bytes_io: int,
    elapsed_sec: float,
    peak_tflops: float = 918.0,
    peak_bw_gbs: float = 1600.0,
):
  ai = flops / max(bytes_io, 1)
  ridge = (peak_tflops * 1e3) / peak_bw_gbs
  gflops_s = (flops / max(elapsed_sec, 1e-9)) / 1e9
  gb_s = (bytes_io / max(elapsed_sec, 1e-9)) / 1e9
  mfu_pct = (gflops_s / (peak_tflops * 1e3)) * 100.0
  bw_pct = (gb_s / peak_bw_gbs) * 100.0
  bound = "COMPUTE-BOUND (MXU)" if ai >= ridge else "MEMORY-BOUND (HBM)"
  print(
      f"  {name:<30} | {elapsed_sec*1e3:7.3f} ms | AI: {ai:6.1f} FLOP/B |"
      f" {gflops_s:7.1f} GFLOPs/s (MFU {mfu_pct:4.1f}%) | {gb_s:6.1f} GB/s (BW"
      f" {bw_pct:4.1f}%) | {bound}"
  )


def run_benchmark():
  print("=" * 116)
  print(
      "ROOFLINE & LLM OPERATOR BENCHMARK (PREFILL MATMUL • DECODE MATMUL •"
      " RMSNORM • SWIGLU • CAUSAL SDPA)"
  )
  print("=" * 116)

  k_dim, n_dim = 1024, 1024
  m_prefill = 1024
  m_decode = 1
  iters = 15

  # 1. PyTorch / TorchTPU Benchmark
  try:
    import torch
    import torch.nn.functional as F
    try:
      import torch_tpu  # noqa: F401
    except ImportError:
      pass

    has_tpu = (
        hasattr(torch, "accelerator")
        and torch.accelerator.is_available()
        and getattr(torch.accelerator.current_accelerator(), "type", None) == "tpu"
    )
    device = torch.device("tpu") if has_tpu else torch.device("cpu")
    sync_fn = torch.accelerator.synchronize if has_tpu else (lambda: None)
    print(f"\n[1] PyTorch / TorchTPU Execution Device: {device}")

    with torch.no_grad():
      x_pre = torch.randn((m_prefill, k_dim), dtype=torch.bfloat16, device=device)
      x_dec = torch.randn((m_decode, k_dim), dtype=torch.bfloat16, device=device)
      w = torch.randn((k_dim, n_dim), dtype=torch.bfloat16, device=device)
      q = torch.randn((2, 8, 512, 64), dtype=torch.bfloat16, device=device)
      k_t = torch.randn((2, 8, 512, 64), dtype=torch.bfloat16, device=device)
      v_t = torch.randn((2, 8, 512, 64), dtype=torch.bfloat16, device=device)

      # Warmup
      _ = x_pre @ w
      _ = x_dec @ w
      _ = F.scaled_dot_product_attention(q, k_t, v_t, is_causal=True)
      sync_fn()

      # Prefill Matmul
      sync_fn()
      t0 = time.perf_counter()
      for _ in range(iters):
        _ = x_pre @ w
      sync_fn()
      dt_pre = (time.perf_counter() - t0) / iters
      compute_roofline_metrics(
          "PyTorch Prefill Matmul (M=1024)",
          2 * m_prefill * k_dim * n_dim,
          2 * (m_prefill * k_dim + k_dim * n_dim + m_prefill * n_dim),
          dt_pre,
      )

      # Decode Matmul
      sync_fn()
      t0 = time.perf_counter()
      for _ in range(iters):
        _ = x_dec @ w
      sync_fn()
      dt_dec = (time.perf_counter() - t0) / iters
      compute_roofline_metrics(
          "PyTorch Decode Matmul  (M=1)",
          2 * m_decode * k_dim * n_dim,
          2 * (m_decode * k_dim + k_dim * n_dim + m_decode * n_dim),
          dt_dec,
      )

      # Causal SDPA (B=2, H=8, S=512, D=64)
      sync_fn()
      t0 = time.perf_counter()
      for _ in range(iters):
        _ = F.scaled_dot_product_attention(q, k_t, v_t, is_causal=True)
      sync_fn()
      dt_sdpa = (time.perf_counter() - t0) / iters
      b_s, h_s, s_len, d_h = q.shape
      flops_sdpa = 4 * b_s * h_s * (s_len**2) * d_h
      bytes_sdpa = 2 * 4 * b_s * h_s * s_len * d_h
      compute_roofline_metrics(
          "PyTorch Causal SDPA    (S=512)", flops_sdpa, bytes_sdpa, dt_sdpa
      )
  except ImportError:
    print("[PyTorch] Not installed.")

  # 2. JAX (@jax.jit) Benchmark
  try:
    import jax
    import jax.numpy as jnp

    print(f"\n[2] JAX (@jax.jit) Backend: {jax.default_backend()}")
    key = jax.random.key(0)
    x_prefill = jax.random.normal(key, (m_prefill, k_dim), dtype=jnp.bfloat16)
    x_decode = jax.random.normal(key, (m_decode, k_dim), dtype=jnp.bfloat16)
    w = jax.random.normal(key, (k_dim, n_dim), dtype=jnp.bfloat16)

    matmul_fn = jax.jit(lambda a, b: a @ b)
    rmsnorm_fn = jax.jit(
        lambda a: a
        * jax.lax.rsqrt(
            jnp.mean(jnp.square(a.astype(jnp.float32)), axis=-1, keepdims=True)
            + 1e-6
        ).astype(a.dtype)
    )
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
    compute_roofline_metrics(
        "JAX Prefill Matmul     (M=1024)",
        2 * m_prefill * k_dim * n_dim,
        2 * (m_prefill * k_dim + k_dim * n_dim + m_prefill * n_dim),
        dt_pre,
    )

    # 2. Decode Matmul (M=1, K=1024, N=1024)
    t0 = time.perf_counter()
    for _ in range(iters):
      matmul_fn(x_decode, w).block_until_ready()
    dt_dec = (time.perf_counter() - t0) / iters
    compute_roofline_metrics(
        "JAX Decode Matmul      (M=1)",
        2 * m_decode * k_dim * n_dim,
        2 * (m_decode * k_dim + k_dim * n_dim + m_decode * n_dim),
        dt_dec,
    )

    # 3. RMSNorm (M=1024, K=1024)
    t0 = time.perf_counter()
    for _ in range(iters):
      rmsnorm_fn(x_prefill).block_until_ready()
    dt_rms = (time.perf_counter() - t0) / iters
    compute_roofline_metrics(
        "JAX Fused RMSNorm      (M=1024)",
        4 * m_prefill * k_dim,
        2 * 2 * m_prefill * k_dim,
        dt_rms,
    )

    # 4. SwiGLU Gate+Up (M=1024, K=1024, N=1024)
    t0 = time.perf_counter()
    for _ in range(iters):
      swiglu_fn(x_prefill, w, w).block_until_ready()
    dt_swi = (time.perf_counter() - t0) / iters
    compute_roofline_metrics(
        "JAX Fused SwiGLU       (M=1024)",
        4 * m_prefill * k_dim * n_dim + 4 * m_prefill * n_dim,
        2 * (m_prefill * k_dim + 2 * k_dim * n_dim + m_prefill * n_dim),
        dt_swi,
    )
  except ImportError:
    print("[JAX] Not installed.")


if __name__ == "__main__":
  run_benchmark()
