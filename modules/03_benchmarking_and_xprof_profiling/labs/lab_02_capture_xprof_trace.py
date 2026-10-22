#!/usr/bin/env python3
"""Module 3, Lab 2: Capturing Step-Annotated XProf Traces in TorchTPU & JAX."""

import os
import tempfile
import time


def run_pytorch_xprof_trace(trace_root: str):
  try:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function
  except ImportError:
    print("[PyTorch] Not installed; skipping PyTorch profiler section.\n")
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

  print("=" * 80)
  print(f"PART 1 (PYTORCH / TORCHTPU): STEP-ANNOTATED PROFILER TRACE ({device})")
  print("=" * 80)

  torch.manual_seed(0)
  w1 = torch.randn((256, 256), device=device) * 0.05
  w2 = torch.randn((256, 256), device=device) * 0.05
  x = torch.randn((128, 256), device=device)

  def forward_block(inp, weight1, weight2):
    with record_function("linear_gelu_1"):
      h = torch.nn.functional.gelu(inp @ weight1)
    with record_function("linear_residual_2"):
      return (h @ weight2) + inp

  compiled_fn = torch.compile(forward_block, backend=backend_name, dynamic=False)

  # 1. Warmup compilation outside profiler window
  t0 = time.perf_counter()
  _ = compiled_fn(x, w1, w2)
  if has_tpu:
    torch.accelerator.synchronize()
  warmup_ms = (time.perf_counter() - t0) * 1e3
  print(f"  Warmup Step 0 (Compile + Execute) : {warmup_ms:.3f} ms")

  # 2. Capture step-annotated trace
  pt_trace_dir = os.path.join(trace_root, "pytorch_trace")
  os.makedirs(pt_trace_dir, exist_ok=True)
  step_times = []
  with profile(activities=[ProfilerActivity.CPU]) as prof:
    for step in range(1, 6):
      ts = time.perf_counter()
      with record_function(f"train_step_{step}"):
        _ = compiled_fn(x, w1, w2)
      if has_tpu:
        torch.accelerator.synchronize()
      prof.step()
      step_times.append((time.perf_counter() - ts) * 1e3)

  for idx, dt in enumerate(step_times, start=1):
    print(f"  Profiled Step {idx} Latency           : {dt:.3f} ms")
  print(f"  Average Steady-State Step Time    : {sum(step_times) / len(step_times):.3f} ms\n")


def run_jax_xprof_trace(trace_root: str):
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed; skipping JAX profiler section.")
    return

  print("=" * 80)
  print(f"PART 2 (JAX): STEP-ANNOTATED XPROF TRACE CAPTURE ({jax.default_backend()})")
  print("=" * 80)

  key = jax.random.key(0)
  k1, k2, k3 = jax.random.split(key, 3)
  w1 = jax.random.normal(k1, (256, 256)) * 0.05
  w2 = jax.random.normal(k2, (256, 256)) * 0.05
  x = jax.random.normal(k3, (128, 256))

  @jax.jit
  def traced_step(w_a, w_b, inp):
    with jax.named_scope("ffn_layer_1"):
      h = jax.nn.gelu(inp @ w_a)
    with jax.named_scope("ffn_layer_2"):
      out = h @ w_b + inp
    return jnp.mean(out ** 2)

  # 1. Warmup outside profiler window
  t0 = time.perf_counter()
  _ = traced_step(w1, w2, x).block_until_ready()
  warmup_ms = (time.perf_counter() - t0) * 1e3
  print(f"  Warmup Step 0 (Includes XLA Compile) : {warmup_ms:.3f} ms")

  # 2. Capture 5 steady-state steps inside jax.profiler.trace
  jax_trace_dir = os.path.join(trace_root, "jax_trace")
  step_times = []
  with jax.profiler.trace(jax_trace_dir):
    for step in range(1, 6):
      t_step = time.perf_counter()
      with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
        loss = traced_step(w1, w2, x).block_until_ready()
      step_times.append((time.perf_counter() - t_step) * 1e3)

  for idx, dt in enumerate(step_times, start=1):
    print(f"  Profiled Step {idx} (StepTraceAnnotation): {dt:.3f} ms | loss={float(loss):.6f}")

  xplane_files = []
  for root, _, files in os.walk(jax_trace_dir):
    for f in files:
      if f.endswith(".xplane.pb") or f.endswith(".gz"):
        xplane_files.append(os.path.join(root, f))

  print(f"\n  Generated XProf Trace Files ({len(xplane_files)} found):")
  for path in xplane_files[:5]:
    size_kb = os.path.getsize(path) / 1024.0
    print(f"    - {path} ({size_kb:.1f} KiB)")


if __name__ == "__main__":
  root_dir = tempfile.mkdtemp(prefix="tpu_lab_xprof_")
  run_pytorch_xprof_trace(root_dir)
  run_jax_xprof_trace(root_dir)
