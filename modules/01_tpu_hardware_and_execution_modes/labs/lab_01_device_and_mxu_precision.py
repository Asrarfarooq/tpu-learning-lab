#!/usr/bin/env python3
'''Module 1, Lab 1: Floating-Point Bit Layouts, MXU Multi-Pass Precision & Tile Alignment.'''

import struct
import time
import numpy as np


def float32_to_bits(val: float) -> str:
  packed = struct.pack(">f", val)
  integer = struct.unpack(">I", packed)[0]
  bits = f"{integer:032b}"
  return f"sign={bits[0]} | exp={bits[1:9]} | mantissa={bits[9:]}"


def task_1_floating_point_bit_layouts():
  print("=" * 78)
  print("TASK 1: IEEE FLOAT32 vs. FLOAT16 vs. BFLOAT16 BIT LAYOUT & DYNAMIC RANGE")
  print("=" * 78)

  sample = 3.14159265
  bits_str = float32_to_bits(sample)
  print(f"Value = {sample}")
  print(f"  float32  (32-bit): {bits_str}")
  print(f"  bfloat16 (16-bit): {bits_str[:38]} (truncates lower 16 mantissa bits)")

  large_val = 1.0e30
  fp32_val = np.float32(large_val)
  with np.errstate(over="ignore"):
    fp16_val = np.float16(large_val)
  print(f"\nDynamic Range Test with value = {large_val:.1e}:")
  print(f"  float32  representation : {fp32_val:.4e} (Preserved)")
  print(f"  float16  representation : {fp16_val}       (OVERFLOWS! Max float16 is 65,504)")
  try:
    import torch
    bf16_val = torch.tensor(large_val, dtype=torch.bfloat16).item()
    print(f"  bfloat16 representation : {bf16_val:.4e} (Preserved! Same 8-bit exponent as float32)")
  except ImportError:
    pass
  print()


def task_2_precision_vs_accuracy_and_speed():
  print("=" * 78)
  print("TASK 2: MATMUL PRECISION MODES (bfloat16 vs. float32 DEFAULT / HIGH / HIGHEST)")
  print("=" * 78)

  n = 512
  flops = 2 * (n ** 3)
  rng = np.random.default_rng(42)
  a_f64 = rng.standard_normal((n, n), dtype=np.float64)
  b_f64 = rng.standard_normal((n, n), dtype=np.float64)
  ref_f64 = a_f64 @ b_f64

  a_f32 = a_f64.astype(np.float32)
  b_f32 = b_f64.astype(np.float32)

  try:
    import torch
    try:
      import torch_tpu  # noqa: F401
      device = torch.device("tpu:0" if hasattr(torch, "tpu") and torch.tpu.is_available() else "cpu")
    except ImportError:
      device = torch.device("cpu")

    print(f"[PyTorch on {device}] Comparing (512x512) Matmul against float64 reference:")
    t_a_bf16 = torch.from_numpy(a_f32).to(device=device, dtype=torch.bfloat16)
    t_b_bf16 = torch.from_numpy(b_f32).to(device=device, dtype=torch.bfloat16)
    out_bf16 = (t_a_bf16 @ t_b_bf16).float().cpu().numpy()
    err_bf16 = float(np.max(np.abs(out_bf16 - ref_f64)))
    print(f"  dtype=bfloat16 (1-pass native)  | Max Abs Error vs fp64: {err_bf16:.6e}")

    t_a_f32 = torch.from_numpy(a_f32).to(device=device)
    t_b_f32 = torch.from_numpy(b_f32).to(device=device)
    for mode in ("medium", "high", "highest"):
      torch.set_float32_matmul_precision(mode)
      out_f32 = (t_a_f32 @ t_b_f32).cpu().numpy()
      err_f32 = float(np.max(np.abs(out_f32 - ref_f64)))
      print(f"  dtype=float32  (mode={mode:<7}) | Max Abs Error vs fp64: {err_f32:.6e}")
  except ImportError:
    pass

  try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    print(f"\n[JAX on {jax.default_backend()}] Comparing (512x512) Matmul across lax.Precision:")
    j_a = jnp.asarray(a_f32)
    j_b = jnp.asarray(b_f32)
    j_a_bf16 = j_a.astype(jnp.bfloat16)
    j_b_bf16 = j_b.astype(jnp.bfloat16)

    bf16_fn = jax.jit(lambda x, y: jnp.matmul(x, y).astype(jnp.float32))
    out_j_bf16 = np.asarray(bf16_fn(j_a_bf16, j_b_bf16).block_until_ready())
    t0 = time.perf_counter()
    for _ in range(20):
      bf16_fn(j_a_bf16, j_b_bf16).block_until_ready()
    dt_bf16 = (time.perf_counter() - t0) / 20.0
    gflops_bf16 = (flops / dt_bf16) / 1e9
    print(f"  bfloat16 inputs                | Max Error: {np.max(np.abs(out_j_bf16 - ref_f64)):.6e} | {dt_bf16*1e3:.3f} ms ({gflops_bf16:.1f} GFLOPs/s)")

    for label, prec in [
        ("Precision.DEFAULT (1-pass)", lax.Precision.DEFAULT),
        ("Precision.HIGH    (3-pass)", lax.Precision.HIGH),
        ("Precision.HIGHEST (6-pass)", lax.Precision.HIGHEST),
    ]:
      fn = jax.jit(lambda x, y, p=prec: jnp.matmul(x, y, precision=p))
      out = np.asarray(fn(j_a, j_b).block_until_ready())
      t0 = time.perf_counter()
      for _ in range(20):
        fn(j_a, j_b).block_until_ready()
      dt = (time.perf_counter() - t0) / 20.0
      gflops = (flops / dt) / 1e9
      print(f"  float32 {label} | Max Error: {np.max(np.abs(out - ref_f64)):.6e} | {dt*1e3:.3f} ms ({gflops:.1f} GFLOPs/s)")
  except ImportError:
    pass
  print()


def task_3_tile_alignment():
  print("=" * 78)
  print("TASK 3: SYSTOLIC ARRAY TILE ALIGNMENT (128-ALIGNED vs. UNALIGNED SHAPES)")
  print("=" * 78)
  for dim in (500, 512, 1000, 1024):
    tiles_128 = ((dim + 127) // 128)
    padded_dim = tiles_128 * 128
    useful_cells = dim * dim
    padded_cells = padded_dim * padded_dim
    efficiency = 100.0 * useful_cells / padded_cells
    print(f"  Matrix ({dim:4d} x {dim:4d}) -> Padded to ({padded_dim:4d} x {padded_dim:4d}) [{tiles_128}x{tiles_128} tiles of 128x128] | Tile Utilization: {efficiency:5.1f}%")


if __name__ == "__main__":
  task_1_floating_point_bit_layouts()
  task_2_precision_vs_accuracy_and_speed()
  task_3_tile_alignment()
