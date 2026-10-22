#!/usr/bin/env python3
"""Recipe 06: OSS TorchTPU 0.2.0 FLUX Diffusion Optimization & DTensor TP / SimpleFSDP Recipes.

Grounded in `//depot/google3/third_party/py/torch_tpu/examples/`:
1. `examples/FLUX_optimization/`:
   - Milestone 1: Replace `einops.rearrange(qkv, "B L (K H D) -> K B L H D")` (which triggers a
     physical `as_strided` copy because `K=3` moves to dim 0) with zero-copy
     `qkv.view(B, L, 3, H, D)` + `torch.unbind(dim=2)`.
   - Milestone 2: Replace 5D `(B, 1, L, D/2, 2, 2)` RoPE rotation matrix with compact 2D `[cos, sin]`
     algebraic RoPE (`x0 * cos - x1 * sin`, `x0 * sin + x1 * cos`), cutting RoPE table HBM traffic
     by 50% and fusing into VPU FMA instructions.
   - Milestone 3 & 4: `torch.compile(backend="tpu")` + Pallas `torch.tpu.ops.splash_sdpa` with
     `block_q=2048, block_kv=2048` (`176.05 ms -> 119.35 ms`, `1.48x` speedup on `v6e-1`).
2. `examples/tensor_parallel/dtensor_wrapped/` & `examples/fsdp/fsdp_to_simple_fsdp/`:
   - Declarative `parallelize_module` with `ColwiseParallel` and `RowwiseParallel`.
   - `simple_fsdp.data_parallel` (`47 ms` vs `155 ms` on `v6e-8` = `3.3x` faster than FSDPv2).
"""

import time

try:
  import torch
  import torch.nn.functional as F
  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False

try:
  import torch_tpu  # noqa: F401
except ImportError:
  pass

HAS_TPU = (
    HAS_TORCH
    and hasattr(torch, "accelerator")
    and torch.accelerator.is_available()
    and getattr(torch.accelerator.current_accelerator(), "type", None) == "tpu"
)


def run_pytorch_flux_benchmark() -> None:
  device = torch.device("tpu") if HAS_TPU else torch.device("cpu")
  dtype = torch.bfloat16
  b, seq_len, num_heads, head_dim = 2, 512, 8, 64
  hidden = num_heads * head_dim

  def sync_device():
    if HAS_TPU:
      torch.accelerator.synchronize()

  def baseline_flux(qkv: torch.Tensor, freqs_matrix: torch.Tensor) -> torch.Tensor:
    rearranged = qkv.view(b, seq_len, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4).contiguous()
    q, k, v = rearranged[0], rearranged[1], rearranged[2]
    q_pairs = q.float().reshape(b, num_heads, seq_len, head_dim // 2, 1, 2)
    k_pairs = k.float().reshape(b, num_heads, seq_len, head_dim // 2, 1, 2)
    q_rot = (
        freqs_matrix[:, :, :, :, :, 0] * q_pairs[..., 0]
        + freqs_matrix[:, :, :, :, :, 1] * q_pairs[..., 1]
    ).reshape(b, num_heads, seq_len, head_dim).to(qkv.dtype)
    k_rot = (
        freqs_matrix[:, :, :, :, :, 0] * k_pairs[..., 0]
        + freqs_matrix[:, :, :, :, :, 1] * k_pairs[..., 1]
    ).reshape(b, num_heads, seq_len, head_dim).to(qkv.dtype)
    return F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=False)

  def optimized_flux(qkv: torch.Tensor, freqs_cos_sin: torch.Tensor) -> torch.Tensor:
    q, k, v = qkv.view(b, seq_len, 3, num_heads, head_dim).unbind(dim=2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    cos, sin = freqs_cos_sin[..., 0], freqs_cos_sin[..., 1]
    q_r = q.float().reshape(b, num_heads, seq_len, head_dim // 2, 2)
    k_r = k.float().reshape(b, num_heads, seq_len, head_dim // 2, 2)
    q0, q1 = q_r[..., 0], q_r[..., 1]
    k0, k1 = k_r[..., 0], k_r[..., 1]
    q_rot = torch.stack((q0 * cos - q1 * sin, q0 * sin + q1 * cos), dim=-1).reshape(q.shape).to(qkv.dtype)
    k_rot = torch.stack((k0 * cos - k1 * sin, k0 * sin + k1 * cos), dim=-1).reshape(k.shape).to(qkv.dtype)
    return F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=False)

  torch.manual_seed(0)
  qkv = torch.randn(b, seq_len, 3 * hidden, device=device, dtype=dtype)
  angles = torch.randn(1, 1, seq_len, head_dim // 2, device=device, dtype=torch.float32)
  cos, sin = torch.cos(angles), torch.sin(angles)
  freqs_matrix = torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)
  freqs_compact = torch.stack([cos, sin], dim=-1)

  with torch.no_grad():
    out_base = baseline_flux(qkv, freqs_matrix)
    out_opt = optimized_flux(qkv, freqs_compact)
    sync_device()

  max_diff = (out_base.float() - out_opt.float()).abs().max().item()
  matrix_bytes = freqs_matrix.numel() * freqs_matrix.element_size()
  compact_bytes = freqs_compact.numel() * freqs_compact.element_size()

  iters = 20
  with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(iters):
      _ = baseline_flux(qkv, freqs_matrix)
    sync_device()
    base_ms = ((time.perf_counter() - t0) / iters) * 1e3

    t1 = time.perf_counter()
    for _ in range(iters):
      _ = optimized_flux(qkv, freqs_compact)
    sync_device()
    opt_ms = ((time.perf_counter() - t1) / iters) * 1e3

  print(f"  - Active Runtime           : PyTorch ({device})")
  print(f"  - RoPE Table HBM Footprint : {matrix_bytes:,} B (2x2 matrix) -> {compact_bytes:,} B (compact [cos,sin], 50% reduction)")
  print(f"  - Numerical Parity Max Diff: {max_diff:.2e}")
  print(f"  - Baseline Latency         : {base_ms:7.3f} ms")
  print(f"  - Optimized Latency        : {opt_ms:7.3f} ms ({base_ms / max(opt_ms, 1e-6):.2f}x speedup)")


def run_jax_flux_verification() -> None:
  import jax
  import jax.numpy as jnp
  import numpy as np

  b, seq_len, num_heads, head_dim = 2, 256, 8, 64
  rng = np.random.default_rng(0)
  q_np = rng.standard_normal((b, num_heads, seq_len, head_dim), dtype=np.float32)
  angles_np = rng.standard_normal((1, 1, seq_len, head_dim // 2), dtype=np.float32)
  cos_np, sin_np = np.cos(angles_np), np.sin(angles_np)

  # 5D 2x2 rotation matrix vs 2D compact [cos, sin]
  freqs_matrix = jnp.asarray(np.stack([np.stack([cos_np, -sin_np], axis=-1), np.stack([sin_np, cos_np], axis=-1)], axis=-2))
  freqs_compact = jnp.asarray(np.stack([cos_np, sin_np], axis=-1))
  q = jnp.asarray(q_np)

  @jax.jit
  def rope_5d_matrix(x, mat):
    pairs = x.reshape(b, num_heads, seq_len, head_dim // 2, 1, 2)
    return (mat[:, :, :, :, :, 0] * pairs[..., 0] + mat[:, :, :, :, :, 1] * pairs[..., 1]).reshape(x.shape)

  @jax.jit
  def rope_2d_algebraic(x, cs):
    c, s = cs[..., 0], cs[..., 1]
    xr = x.reshape(b, num_heads, seq_len, head_dim // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    return jnp.stack([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1).reshape(x.shape)

  out_5d = rope_5d_matrix(q, freqs_matrix).block_until_ready()
  out_2d = rope_2d_algebraic(q, freqs_compact).block_until_ready()
  max_diff = float(jnp.max(jnp.abs(out_5d - out_2d)))
  print(f"  - Active Runtime           : JAX ({jax.default_backend()})")
  print(f"  - RoPE Table HBM Footprint : {freqs_matrix.nbytes:,} B (2x2 matrix) -> {freqs_compact.nbytes:,} B (compact [cos,sin], 50% reduction)")
  print(f"  - 5D Matrix vs 2D Algebraic RoPE Max Abs Diff: {max_diff:.2e}")


def main() -> None:
  print("=" * 88)
  print("OSS TorchTPU 0.2.0 Recipe 06: FLUX Diffusion Optimization & DTensor TP / SimpleFSDP")
  print("=" * 88)
  print("\n[1] FLUX Attention + RoPE Optimization (examples/FLUX_optimization):")
  if HAS_TORCH:
    run_pytorch_flux_benchmark()
  else:
    run_jax_flux_verification()

  print("  - Full v6e-1 (B=2, S=32,768, D=1024, bfloat16) Measured Ablation (examples/FLUX_optimization/README.md):")
  print("      1. Baseline Reference (Eager)                                  : 176.05 ms (1.00x)")
  print("      2. Zero-Copy Views Only (Eager)                                : 175.39 ms (1.00x)")
  print("      3. Compile Only (Unoptimized Graph)                            : 157.33 ms (1.12x)")
  print("      4. Zero-Copy Views + Compile                                   : 156.56 ms (1.12x)")
  print("      5. Zero-Copy + Fused 2D RoPE + Compile (Standard SDPA)         : 151.27 ms (1.16x)")
  print("      6. Standalone SplashAttention (Eager)                          : 147.50 ms (1.19x)")
  print("      7. Zero-Copy + Fused 2D RoPE + SplashAttention (Eager)         : 153.60 ms (1.15x)")
  print("      8. Fully Optimized (Zero-Copy + 2D RoPE + SplashAttn + Compile): 119.35 ms (1.48x speedup!)")

  print("\n[2] Distributed FSDPv2 vs. SimpleFSDP Benchmark Summary on v6e-8 (examples/fsdp/fsdp_to_simple_fsdp):")
  print("  - FSDPv2 Eager (`fully_shard`)                 : ~270.0 ms / step")
  print("  - FSDPv2 + Per-DecoderLayer `torch.compile`    : ~155.0 ms / step")
  print("  - SimpleFSDP (`simple_fsdp.data_parallel`)     :  ~47.0 ms / step (3.30x faster than compiled FSDPv2!)")
  print("  - Key Rule: SimpleFSDP uses SUM gradient reduction; scale loss by `(loss / world_size).backward()`.")


if __name__ == "__main__":
  main()
