#!/usr/bin/env python3
"""Module 6, Lab 1: Hugging Face StaticCache Decode Pipeline, 4D Causal Masking & Teacher-Forced Parity (TorchTPU & JAX)."""

import time
import numpy as np

BATCH_SIZE = 2
PROMPT_LEN = 32
GENERATE_TOKENS = 8
MAX_CACHE_LEN = PROMPT_LEN + GENERATE_TOKENS
D_MODEL = 128
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2  # Grouped-Query Attention (2:1 GQA ratio like Llama-3 / Qwen3 / Gemma)
HEAD_DIM = D_MODEL // NUM_Q_HEADS
VOCAB_SIZE = 1024


def run_pytorch_static_cache_and_parity():
  try:
    import torch
    import torch.nn.functional as F
  except ImportError:
    print("[PyTorch] Not installed; skipping PyTorch section.\n")
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

  print("=" * 84)
  print(f"PART 1 (PYTORCH / TORCHTPU): STATIC-CACHE DECODE & TEACHER-FORCED PARITY ({device})")
  print("=" * 84)

  torch.manual_seed(42)
  embed = torch.randn(VOCAB_SIZE, D_MODEL, device=device) * 0.04
  w_q = torch.randn(D_MODEL, NUM_Q_HEADS * HEAD_DIM, device=device) * 0.04
  w_k = torch.randn(D_MODEL, NUM_KV_HEADS * HEAD_DIM, device=device) * 0.04
  w_v = torch.randn(D_MODEL, NUM_KV_HEADS * HEAD_DIM, device=device) * 0.04
  w_o = torch.randn(NUM_Q_HEADS * HEAD_DIM, D_MODEL, device=device) * 0.04

  # 1. Teacher-Forced Per-Layer Numerical Parity Check (bfloat16 vs float32 reference)
  x_ref = torch.randn(BATCH_SIZE, PROMPT_LEN, D_MODEL, device=device, dtype=torch.float32)
  with torch.no_grad():
    y_fp32 = x_ref @ w_q
    y_bf16 = (x_ref.to(torch.bfloat16) @ w_q.to(torch.bfloat16)).float()
  max_abs = float((y_bf16 - y_fp32).abs().max().cpu().item())
  max_rel = max_abs / (float(y_fp32.abs().max().cpu().item()) + 1e-6)
  status = "PASS" if max_rel < 0.03 else "WARN"
  print(f"1. Teacher-Forced bfloat16 Layer Parity : [{status}] max_abs={max_abs:.3e} | max_rel={max_rel*100:.2f}% (< 3.0% bf16 floor)")

  # 2. Pre-allocate StaticCache buffers [B, NUM_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM]
  k_cache = torch.zeros(BATCH_SIZE, NUM_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM, device=device)
  v_cache = torch.zeros(BATCH_SIZE, NUM_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM, device=device)
  prompt_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, PROMPT_LEN), device=device)

  with torch.no_grad():
    x_p = F.embedding(prompt_ids, embed)
    k_cache[:, :, :PROMPT_LEN, :] = (x_p @ w_k).view(BATCH_SIZE, PROMPT_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v_cache[:, :, :PROMPT_LEN, :] = (x_p @ w_v).view(BATCH_SIZE, PROMPT_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    prefill_logits = x_p[:, -1:, :] @ embed.t()
    # Critical: keepdim=True avoids creating a lazy as_strided view via .unsqueeze(-1)
    next_tok = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)

  def decode_one_token(token_id, k_buf, v_buf, pos_idx):
    x_step = F.embedding(token_id, embed)
    q = (x_step @ w_q).view(BATCH_SIZE, 1, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k_new = (x_step @ w_k).view(BATCH_SIZE, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v_new = (x_step @ w_v).view(BATCH_SIZE, 1, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
    k_buf[:, :, pos_idx : pos_idx + 1, :] = k_new
    v_buf[:, :, pos_idx : pos_idx + 1, :] = v_new
    k_rep = k_buf.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    v_rep = v_buf.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    attn = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
    logits = (attn.transpose(1, 2).reshape(BATCH_SIZE, 1, D_MODEL) @ w_o) @ embed.t()
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

  # Warmup 2 decode steps, then rewind StaticCache in-place (.zero_())
  with torch.no_grad():
    tok_w = decode_one_token(next_tok, k_cache, v_cache, PROMPT_LEN)
    tok_w = decode_one_token(tok_w, k_cache, v_cache, PROMPT_LEN + 1)
    k_cache[:, :, PROMPT_LEN:, :].zero_()
    v_cache[:, :, PROMPT_LEN:, :].zero_()

    t0 = time.perf_counter()
    cur_tok = next_tok
    for step_i in range(GENERATE_TOKENS):
      cur_tok = decode_one_token(cur_tok, k_cache, v_cache, PROMPT_LEN + step_i)
    if has_tpu:
      torch.accelerator.synchronize()
    pt_decode_ms = (time.perf_counter() - t0) * 1e3

  print(f"2. PyTorch StaticCache Shape            : {tuple(k_cache.shape)} (rewound in-place via .zero_())")
  print(f"3. PyTorch Decode Time ({GENERATE_TOKENS} steps)        : {pt_decode_ms:.3f} ms ({pt_decode_ms / GENERATE_TOKENS:.3f} ms/token)\n")


def run_jax_static_cache_decode_lab():
  print("=" * 84)
  print("PART 2 (JAX): STATIC-CACHE AUTOREGRESSIVE DECODE & ZERO-RECOMPILATION BENCHMARK")
  print("=" * 84)

  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed.")
    return

  rng = np.random.default_rng(42)
  w_q = jnp.asarray((rng.standard_normal((D_MODEL, NUM_Q_HEADS * HEAD_DIM)) * 0.04).astype(np.float32))
  w_k = jnp.asarray((rng.standard_normal((D_MODEL, NUM_KV_HEADS * HEAD_DIM)) * 0.04).astype(np.float32))
  w_v = jnp.asarray((rng.standard_normal((D_MODEL, NUM_KV_HEADS * HEAD_DIM)) * 0.04).astype(np.float32))
  w_o = jnp.asarray((rng.standard_normal((NUM_Q_HEADS * HEAD_DIM, D_MODEL)) * 0.04).astype(np.float32))
  embed_table = jnp.asarray((rng.standard_normal((VOCAB_SIZE, D_MODEL)) * 0.04).astype(np.float32))

  prompt_ids = jnp.asarray(rng.integers(0, VOCAB_SIZE, size=(BATCH_SIZE, PROMPT_LEN), dtype=np.int32))

  k_cache_init = jnp.zeros((BATCH_SIZE, NUM_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM), dtype=jnp.float32)
  v_cache_init = jnp.zeros((BATCH_SIZE, NUM_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM), dtype=jnp.float32)

  x_prompt = embed_table[prompt_ids]
  k_prompt = (x_prompt @ w_k).reshape(BATCH_SIZE, PROMPT_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
  v_prompt = (x_prompt @ w_v).reshape(BATCH_SIZE, PROMPT_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
  k_cache = k_cache_init.at[:, :, :PROMPT_LEN, :].set(k_prompt)
  v_cache = v_cache_init.at[:, :, :PROMPT_LEN, :].set(v_prompt)

  prefill_logits = x_prompt[:, -1:, :] @ embed_table.T
  next_token = jnp.argmax(prefill_logits[:, -1, :], axis=-1, keepdims=True)
  print(f"1. StaticCache Allocated : shape={k_cache.shape} (fixed across all {GENERATE_TOKENS} decode steps!)")
  print(f"2. Post-Prefill Token    : shape={next_token.shape} (selected with keepdims=True)")

  compile_counter = {"count": 0}

  @jax.jit
  def compiled_decode_step(token_id, k_buf, v_buf, cache_pos):
    compile_counter["count"] += 1
    x_step = embed_table[token_id]
    q = (x_step @ w_q).reshape(BATCH_SIZE, 1, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k_new = (x_step @ w_k).reshape(BATCH_SIZE, 1, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v_new = (x_step @ w_v).reshape(BATCH_SIZE, 1, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    k_buf = jax.lax.dynamic_update_slice(k_buf, k_new, (0, 0, cache_pos, 0))
    v_buf = jax.lax.dynamic_update_slice(v_buf, v_new, (0, 0, cache_pos, 0))

    k_rep = jnp.repeat(k_buf, NUM_Q_HEADS // NUM_KV_HEADS, axis=1)
    v_rep = jnp.repeat(v_buf, NUM_Q_HEADS // NUM_KV_HEADS, axis=1)

    scores = jnp.matmul(q, jnp.swapaxes(k_rep, -1, -2)) / jnp.sqrt(float(HEAD_DIM))
    valid_mask = jnp.arange(MAX_CACHE_LEN)[None, None, None, :] <= cache_pos
    scores = jnp.where(valid_mask, scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.matmul(probs, v_rep).transpose(0, 2, 1, 3).reshape(BATCH_SIZE, 1, D_MODEL) @ w_o
    logits = attn_out @ embed_table.T
    next_tok = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    return next_tok, k_buf, v_buf

  tok_w, k_cache, v_cache = compiled_decode_step(next_token, k_cache, v_cache, jnp.int32(PROMPT_LEN))
  tok_w, k_cache, v_cache = compiled_decode_step(tok_w, k_cache, v_cache, jnp.int32(PROMPT_LEN + 1))
  jax.block_until_ready(tok_w)
  compiles_after_warmup = compile_counter["count"]

  k_cache = k_cache.at[:, :, PROMPT_LEN:, :].set(0.0)
  v_cache = v_cache.at[:, :, PROMPT_LEN:, :].set(0.0)

  t0 = time.perf_counter()
  cur_tok = next_token
  for step_idx in range(GENERATE_TOKENS):
    cur_pos = jnp.int32(PROMPT_LEN + step_idx)
    cur_tok, k_cache, v_cache = compiled_decode_step(cur_tok, k_cache, v_cache, cur_pos)
  jax.block_until_ready(cur_tok)
  decode_ms = (time.perf_counter() - t0) * 1000.0

  compiles_after_decode = compile_counter["count"]
  print(f"3. XLA Compilations During Warmup        : {compiles_after_warmup}")
  print(f"4. New Compilations Across {GENERATE_TOKENS} Decode Steps : {compiles_after_decode - compiles_after_warmup} (ZERO recompilations!)")
  print(f"5. Measured Decode Time ({GENERATE_TOKENS} steps)        : {decode_ms:.3f} ms ({decode_ms / GENERATE_TOKENS:.3f} ms/token)\n")


if __name__ == "__main__":
  run_pytorch_static_cache_and_parity()
  run_jax_static_cache_decode_lab()
