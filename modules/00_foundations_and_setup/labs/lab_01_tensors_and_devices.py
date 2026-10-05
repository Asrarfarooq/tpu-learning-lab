#!/usr/bin/env python3
'''Module 0, Lab 1: Tensors, Memory Layouts, Matrix Multiplication & RNG (TorchTPU & JAX).'''

import sys


def task_1_hardware_discovery():
  print("=" * 76)
  print("TASK 1: HARDWARE & RUNTIME DISCOVERY")
  print("=" * 76)
  print(f"Python executable : {sys.executable}")
  print(f"Python version    : {sys.version.split()[0]}")

  torch_ctx = {"available": False, "device": "cpu"}
  try:
    import torch
    has_tpu = False
    try:
      import torch_tpu  # noqa: F401
      has_tpu = hasattr(torch, "tpu") and torch.tpu.is_available()
    except ImportError:
      pass

    dev_str = "tpu:0" if has_tpu else "cpu"
    dev_count = torch.tpu.device_count() if has_tpu else 1
    torch_ctx = {"available": True, "device": dev_str, "has_tpu": has_tpu}
    print(f"[PyTorch] version={torch.__version__} | device={dev_str} | device_count={dev_count}")
    if has_tpu and hasattr(torch, "accelerator") and hasattr(torch.accelerator, "get_memory_info"):
      free_b, total_b = torch.accelerator.get_memory_info(dev_str)
      print(f"[PyTorch] TPU HBM Free: {free_b / (1024**3):.2f} GiB / Total: {total_b / (1024**3):.2f} GiB")
  except ImportError:
    print("[PyTorch] Not installed in this environment.")

  jax_ctx = {"available": False, "backend": "cpu"}
  try:
    import jax
    devs = jax.devices()
    jax_ctx = {"available": True, "backend": jax.default_backend(), "devices": devs}
    print(f"[JAX]     version={jax.__version__} | backend={jax.default_backend()} | device_count={len(devs)}")
    print(f"[JAX]     devices={[str(d) for d in devs]}")
  except ImportError:
    print("[JAX]     Not installed in this environment.")

  print()
  return torch_ctx, jax_ctx


def task_2_tensor_anatomy(torch_ctx, jax_ctx):
  print("=" * 76)
  print("TASK 2: TENSOR ANATOMY (RANK, SHAPE, STRIDES, DTYPE & HBM FOOTPRINT)")
  print("=" * 76)

  rows, cols = 1024, 1024
  if torch_ctx["available"]:
    import torch
    device = torch.device(torch_ctx["device"])
    x_fp32 = torch.ones((rows, cols), dtype=torch.float32, device=device)
    x_bf16 = torch.ones((rows, cols), dtype=torch.bfloat16, device=device)

    fp32_mib = (x_fp32.numel() * x_fp32.element_size()) / (1024 ** 2)
    bf16_mib = (x_bf16.numel() * x_bf16.element_size()) / (1024 ** 2)
    print(f"[PyTorch] x_fp32: shape={tuple(x_fp32.shape)}, rank={x_fp32.ndim}, strides={x_fp32.stride()}, dtype={x_fp32.dtype}, size={fp32_mib:.2f} MiB")
    print(f"[PyTorch] x_bf16: shape={tuple(x_bf16.shape)}, rank={x_bf16.ndim}, strides={x_bf16.stride()}, dtype={x_bf16.dtype}, size={bf16_mib:.2f} MiB")

  if jax_ctx["available"]:
    import jax.numpy as jnp
    j_fp32 = jnp.ones((rows, cols), dtype=jnp.float32)
    j_bf16 = jnp.ones((rows, cols), dtype=jnp.bfloat16)
    print(f"[JAX]     j_fp32: shape={j_fp32.shape}, rank={j_fp32.ndim}, dtype={j_fp32.dtype}, size={j_fp32.nbytes / (1024**2):.2f} MiB")
    print(f"[JAX]     j_bf16: shape={j_bf16.shape}, rank={j_bf16.ndim}, dtype={j_bf16.dtype}, size={j_bf16.nbytes / (1024**2):.2f} MiB")
  print()


def task_3_matrix_multiplication_and_flops(torch_ctx, jax_ctx):
  print("=" * 76)
  print("TASK 3: MATRIX MULTIPLICATION ([M, K] @ [K, N]) & FLOP ACCOUNTING")
  print("=" * 76)

  m, k, n = 4, 8, 3
  flops = 2 * m * n * k
  print(f"Dimensions: [M={m}, K={k}] @ [K={k}, N={n}] -> Output [M={m}, N={n}]")
  print(f"Exact FLOPs required (2 * M * N * K): {flops} FLOPs")

  t_out = None
  j_out = None

  if torch_ctx["available"]:
    import torch
    device = torch.device(torch_ctx["device"])
    x_t = torch.arange(m * k, dtype=torch.float32, device=device).reshape(m, k) * 0.1
    w_t = torch.arange(k * n, dtype=torch.float32, device=device).reshape(k, n) * 0.2
    y_t = x_t @ w_t
    t_out = y_t.cpu()
    print(f"[PyTorch] Output Y shape={tuple(y_t.shape)}:\n{t_out}")

  if jax_ctx["available"]:
    import jax.numpy as jnp
    x_j = jnp.arange(m * k, dtype=jnp.float32).reshape(m, k) * 0.1
    w_j = jnp.arange(k * n, dtype=jnp.float32).reshape(k, n) * 0.2
    y_j = x_j @ w_j
    j_out = y_j
    print(f"[JAX]     Output Y shape={y_j.shape}:\n{j_out}")

  if t_out is not None and j_out is not None:
    import numpy as np
    max_diff = float(np.max(np.abs(t_out.numpy() - np.asarray(j_out))))
    print(f"[Parity]  Max absolute difference between PyTorch and JAX: {max_diff:.2e}")
  print()


def task_4_mutability_and_rng(torch_ctx, jax_ctx):
  print("=" * 76)
  print("TASK 4: MUTABILITY & STATELESS PRNG SPLITTING")
  print("=" * 76)

  if torch_ctx["available"]:
    import torch
    device = torch.device(torch_ctx["device"])
    torch.manual_seed(42)
    a = torch.zeros((2, 3), device=device)
    a[0, 2] = 99.0
    print(f"[PyTorch] In-place mutation a[0, 2] = 99.0:\n{a.cpu()}")

  if jax_ctx["available"]:
    import jax
    import jax.numpy as jnp
    arr = jnp.zeros((2, 3))
    try:
      arr[0, 2] = 99.0
    except TypeError as exc:
      print(f"[JAX]     In-place mutation arr[0, 2] = 99.0 blocked: {exc}")

    arr_new = arr.at[0, 2].set(99.0)
    print(f"[JAX]     Immutable update arr.at[0, 2].set(99.0):\n{arr_new}")

    key = jax.random.key(42)
    same_1 = jax.random.normal(key, (3,))
    same_2 = jax.random.normal(key, (3,))
    print(f"[JAX]     Reusing unsplit key produces identical values: diff={float(jnp.max(jnp.abs(same_1 - same_2))):.1f}")

    key, sub1, sub2 = jax.random.split(key, 3)
    diff_1 = jax.random.normal(sub1, (3,))
    diff_2 = jax.random.normal(sub2, (3,))
    print(f"[JAX]     Splitting key produces independent draws:\n          sub1 -> {diff_1}\n          sub2 -> {diff_2}")


if __name__ == "__main__":
  t_ctx, j_ctx = task_1_hardware_discovery()
  task_2_tensor_anatomy(t_ctx, j_ctx)
  task_3_matrix_multiplication_and_flops(t_ctx, j_ctx)
  task_4_mutability_and_rng(t_ctx, j_ctx)
