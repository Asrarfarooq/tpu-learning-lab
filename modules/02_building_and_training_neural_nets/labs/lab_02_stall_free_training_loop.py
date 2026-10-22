#!/usr/bin/env python3
'''Module 2, Lab 2: Eliminating Host-Device Stalls (Detached Loss Accumulation & jax.lax.scan).'''

import time


def benchmark_pytorch_host_stalls():
  try:
    import torch
  except ImportError:
    print("[PyTorch] Not installed; skipping.")
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
  print("=" * 78)
  print(f"PART 1: PYTORCH PER-STEP .item() STALL vs. DETACHED ACCUMULATION ({device})")
  print("=" * 78)

  torch.manual_seed(0)
  x = torch.randn((256, 256), device=device)
  y = torch.randn((256, 256), device=device)
  w = torch.randn((256, 256), device=device, requires_grad=True)

  def train_step(inp, tgt, weight):
    pred = torch.tanh(inp @ weight)
    loss = ((pred - tgt) ** 2).mean()
    loss.backward()
    with torch.no_grad():
      weight -= 0.01 * weight.grad
    weight.grad = None
    return loss

  # Warmup
  _ = train_step(x, y, w).item()

  # Anti-pattern: Syncing loss.item() on every step
  t0 = time.perf_counter()
  for _ in range(40):
    loss = train_step(x, y, w)
    _ = loss.item()  # Host blocks waiting for device on every step!
  stalled_ms = (time.perf_counter() - t0) * 1e3

  # Best Practice: Accumulate loss.detach() on device, sync once at step 40
  t0 = time.perf_counter()
  running_loss = torch.zeros((), device=device)
  for _ in range(40):
    loss = train_step(x, y, w)
    running_loss += loss.detach()
  _ = (running_loss / 40.0).item()
  async_ms = (time.perf_counter() - t0) * 1e3

  print(f"  [Anti-Pattern: Per-Step .item() Sync] 40 Steps Total: {stalled_ms:7.2f} ms")
  print(f"  [Best Practice: Detached Accumulator] 40 Steps Total: {async_ms:7.2f} ms")

  # Demonstrate TorchTPU Memory Model: UntypedStorage sharing across views
  base_t = torch.randn((64, 128), device=device)
  view_t = base_t.transpose(0, 1)
  shares_storage = base_t.untyped_storage().data_ptr() == view_t.untyped_storage().data_ptr()
  print(f"  [TorchTPU Memory Model] View shares UntypedStorage with base tensor: {shares_storage}\n")


def benchmark_jax_lax_scan():
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed; skipping.")
    return

  print("=" * 78)
  print(f"PART 2: JAX PYTHON LOOP vs. FUSED jax.lax.scan ({jax.default_backend()})")
  print("=" * 78)

  key = jax.random.key(0)
  k1, k2, k3 = jax.random.split(key, 3)
  w_init = jax.random.normal(k1, (256, 256))
  xs = jax.random.normal(k2, (40, 64, 256))
  ys = jax.random.normal(k3, (40, 64, 256))

  @jax.jit
  def single_step(w, batch_x, batch_y):
    loss_val, grad = jax.value_and_grad(lambda weight: jnp.mean((jnp.tanh(batch_x @ weight) - batch_y) ** 2))(w)
    return w - 0.01 * grad, loss_val

  @jax.jit
  def scanned_40_steps(w0, all_x, all_y):
    def body(w_carry, xy_pair):
      bx, by = xy_pair
      return single_step(w_carry, bx, by)
    final_w, losses = jax.lax.scan(body, w0, (all_x, all_y))
    return final_w, jnp.mean(losses)

  # Warmup both
  w_tmp, l_tmp = single_step(w_init, xs[0], ys[0])
  _ = float(l_tmp)
  w_scan, l_scan = scanned_40_steps(w_init, xs, ys)
  _ = float(l_scan)

  # 1. Python loop syncing float(loss) every step
  t0 = time.perf_counter()
  w_curr = w_init
  for i in range(40):
    w_curr, l_val = single_step(w_curr, xs[i], ys[i])
    _ = float(l_val)
  py_loop_ms = (time.perf_counter() - t0) * 1e3

  # 2. Single jax.lax.scan call executing all 40 steps inside XLA
  t0 = time.perf_counter()
  w_final, mean_loss = scanned_40_steps(w_init, xs, ys)
  _ = float(mean_loss)
  scan_ms = (time.perf_counter() - t0) * 1e3

  print(f"  [Python Loop + Per-Step float(loss)] 40 Steps Total: {py_loop_ms:7.2f} ms")
  print(f"  [Fused jax.lax.scan (1 XLA Launch)]  40 Steps Total: {scan_ms:7.2f} ms")
  print(f"  [Speedup from jax.lax.scan]          {py_loop_ms / max(scan_ms, 1e-6):.2f}x faster\n")


if __name__ == "__main__":
  benchmark_pytorch_host_stalls()
  benchmark_jax_lax_scan()
