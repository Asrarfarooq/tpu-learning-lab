#!/usr/bin/env python3
"""Module 3, Lab 2: Capturing Step-Annotated XProf Hardware Traces."""

from pathlib import Path
import tempfile


def capture_jax_xprof():
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed; skipping.")
    return

  trace_dir = Path(tempfile.mkdtemp(prefix="xprof_jax_trace_"))
  print("=" * 78)
  print(f"CAPTURING STEP-ANNOTATED XPROF TRACE INTO: {trace_dir}")
  print("=" * 78)

  key = jax.random.key(0)
  x = jax.random.normal(key, (512, 512), dtype=jnp.bfloat16)
  w = jax.random.normal(key, (512, 512), dtype=jnp.bfloat16) * 0.02

  @jax.jit
  def annotated_step(inp, weight):
    with jax.named_scope("ffn_projection"):
      h = inp @ weight
    with jax.named_scope("gelu_activation"):
      act = jax.nn.gelu(h)
    with jax.named_scope("loss_reduction"):
      return jnp.mean(jnp.square(act))

  # Warmup outside profiler so XLA compilation is excluded from the trace
  annotated_step(x, w).block_until_ready()

  with jax.profiler.trace(str(trace_dir)):
    for step_idx in range(1, 6):
      with jax.profiler.StepTraceAnnotation("train_step", step_num=step_idx):
        loss = annotated_step(x, w)
        loss.block_until_ready()
        print(f"  Profiled Step {step_idx} | Loss={float(loss):.6f}")

  files = list(trace_dir.rglob("*"))
  print("\nGenerated XProf Trace Files (.xplane.pb):")
  for f in files:
    if f.is_file():
      print(f"  - {f} ({f.stat().st_size:,} bytes)")
  print("\nTo view in TensorBoard on TPU VM: tensorboard --logdir=" + str(trace_dir) + " --port=6006")


if __name__ == "__main__":
  capture_jax_xprof()
