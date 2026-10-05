#!/usr/bin/env python3
'''Module 5, Lab 2: Capstone End-to-End Transformer Decoder Benchmark (TorchTPU vs. JAX).'''

import math
import time
import numpy as np

BATCH = 8
SEQ_LEN = 128
D_MODEL = 256
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
D_FF = 512
ITERS = 15


def estimate_decoder_layer_flops(b: int, s: int, d: int, df: int) -> int:
  # QKV proj (6*b*s*d^2) + Out proj (2*b*s*d^2) + Attention QK^T & AV (4*b*s^2*d) + SwiGLU (6*b*s*d*df)
  return (8 * b * s * (d ** 2)) + (4 * b * (s ** 2) * d) + (6 * b * s * d * df)


def benchmark_pytorch_transformer():
  try:
    import torch
    import torch.nn as nn
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

  class DecoderBlock(nn.Module):
    def __init__(self):
      super().__init__()
      self.wqkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
      self.wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
      self.w_gate = nn.Linear(D_MODEL, D_FF, bias=False)
      self.w_up = nn.Linear(D_MODEL, D_FF, bias=False)
      self.w_down = nn.Linear(D_FF, D_MODEL, bias=False)

    def rmsnorm(self, x):
      return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)

    def forward(self, x):
      b, s, d = x.shape
      h = self.rmsnorm(x)
      qkv = self.wqkv(h).reshape(b, s, 3, N_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
      q, k, v = qkv[0], qkv[1], qkv[2]
      attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
      attn_out = self.wo(attn.transpose(1, 2).reshape(b, s, d))
      x = x + attn_out
      h2 = self.rmsnorm(x)
      ffn = self.w_down(F.silu(self.w_gate(h2)) * self.w_up(h2))
      return x + ffn

  torch.manual_seed(0)
  model = DecoderBlock().to(device=device)
  compiled_model = torch.compile(model, backend=backend_name, dynamic=False)
  x = torch.randn((BATCH, SEQ_LEN, D_MODEL), device=device)

  # Warmup
  _ = compiled_model(x)
  if has_tpu:
    torch.tpu.synchronize()

  t0 = time.perf_counter()
  for _ in range(ITERS):
    _ = compiled_model(x)
    if has_tpu:
      torch.tpu.synchronize()
  dt = (time.perf_counter() - t0) / ITERS

  flops = estimate_decoder_layer_flops(BATCH, SEQ_LEN, D_MODEL, D_FF)
  tokens_per_sec = (BATCH * SEQ_LEN) / dt
  gflops_per_sec = (flops / dt) / 1e9

  print("=" * 78)
  print(f"PYTORCH TRANSFORMER DECODER BLOCK (device={device}, backend={backend_name})")
  print("=" * 78)
  print(f"  Batch={BATCH}, Seq={SEQ_LEN}, D_Model={D_MODEL}, Heads={N_HEADS}, D_FF={D_FF}")
  print(f"  Avg Step Latency : {dt * 1e3:8.3f} ms")
  print(f"  Throughput       : {tokens_per_sec:8.1f} tokens/sec | {gflops_per_sec:7.2f} GFLOPs/sec\n")


def benchmark_jax_transformer():
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed; skipping.")
    return

  key = jax.random.key(0)
  keys = jax.random.split(key, 6)
  scale = 1.0 / math.sqrt(D_MODEL)
  params = {
      "wqkv": jax.random.normal(keys[0], (D_MODEL, 3 * D_MODEL)) * scale,
      "wo": jax.random.normal(keys[1], (D_MODEL, D_MODEL)) * scale,
      "w_gate": jax.random.normal(keys[2], (D_MODEL, D_FF)) * scale,
      "w_up": jax.random.normal(keys[3], (D_MODEL, D_FF)) * scale,
      "w_down": jax.random.normal(keys[4], (D_FF, D_MODEL)) * scale,
  }
  x = jax.random.normal(keys[5], (BATCH, SEQ_LEN, D_MODEL))

  def rmsnorm(t):
    return t * jax.lax.rsqrt(jnp.mean(jnp.square(t.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6).astype(t.dtype)

  @jax.jit
  def decoder_forward(p, inp):
    b, s, d = inp.shape
    h = rmsnorm(inp)
    qkv = (h @ p["wqkv"]).reshape(b, s, 3, N_HEADS, HEAD_DIM)
    q = jnp.transpose(qkv[:, :, 0], (0, 2, 1, 3))
    k = jnp.transpose(qkv[:, :, 1], (0, 2, 1, 3))
    v = jnp.transpose(qkv[:, :, 2], (0, 2, 1, 3))

    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(HEAD_DIM)
    mask = jnp.tril(jnp.ones((s, s), dtype=jnp.bool_))
    scores = jnp.where(mask, scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1)
    attn = jnp.matmul(probs, v)
    attn_out = jnp.transpose(attn, (0, 2, 1, 3)).reshape(b, s, d) @ p["wo"]

    x_mid = inp + attn_out
    h2 = rmsnorm(x_mid)
    ffn = (jax.nn.silu(h2 @ p["w_gate"]) * (h2 @ p["w_up"])) @ p["w_down"]
    return x_mid + ffn

  # Warmup
  decoder_forward(params, x).block_until_ready()

  t0 = time.perf_counter()
  for _ in range(ITERS):
    decoder_forward(params, x).block_until_ready()
  dt = (time.perf_counter() - t0) / ITERS

  flops = estimate_decoder_layer_flops(BATCH, SEQ_LEN, D_MODEL, D_FF)
  tokens_per_sec = (BATCH * SEQ_LEN) / dt
  gflops_per_sec = (flops / dt) / 1e9

  print("=" * 78)
  print(f"JAX TRANSFORMER DECODER BLOCK (@jax.jit on backend={jax.default_backend()})")
  print("=" * 78)
  print(f"  Batch={BATCH}, Seq={SEQ_LEN}, D_Model={D_MODEL}, Heads={N_HEADS}, D_FF={D_FF}")
  print(f"  Avg Step Latency : {dt * 1e3:8.3f} ms")
  print(f"  Throughput       : {tokens_per_sec:8.1f} tokens/sec | {gflops_per_sec:7.2f} GFLOPs/sec\n")


if __name__ == "__main__":
  benchmark_pytorch_transformer()
  benchmark_jax_transformer()
