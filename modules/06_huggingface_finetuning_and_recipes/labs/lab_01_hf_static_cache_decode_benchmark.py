#!/usr/bin/env python3
'''Module 6, Lab 1: Hugging Face StaticCache Decode Pipeline & 4D Causal Masking.'''

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


def run_static_cache_decode_lab():
  print("=" * 80)
  print("HUGGING FACE STATIC-CACHE AUTOREGRESSIVE DECODE & GQA BENCHMARK")
  print("=" * 80)

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

  # 1. Allocate fixed-shape StaticCache buffers [B, num_kv_heads, MAX_CACHE_LEN, head_dim]
  k_cache_init = jnp.zeros((BATCH_SIZE, NUM_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM), dtype=jnp.float32)
  v_cache_init = jnp.zeros((BATCH_SIZE, NUM_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM), dtype=jnp.float32)

  # 2. Eager Prefill over [B, PROMPT_LEN]
  x_prompt = embed_table[prompt_ids]
  k_prompt = (x_prompt @ w_k).reshape(BATCH_SIZE, PROMPT_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
  v_prompt = (x_prompt @ w_v).reshape(BATCH_SIZE, PROMPT_LEN, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
  k_cache = k_cache_init.at[:, :, :PROMPT_LEN, :].set(k_prompt)
  v_cache = v_cache_init.at[:, :, :PROMPT_LEN, :].set(v_prompt)

  # Select first token with keepdims=True so shape is [B, 1] without lazy view overhead
  prefill_logits = x_prompt[:, -1:, :] @ embed_table.T
  next_token = jnp.argmax(prefill_logits[:, -1, :], axis=-1, keepdims=True)
  print(f"1. StaticCache Allocated : shape={k_cache.shape} (fixed across all {GENERATE_TOKENS} decode steps!)")
  print(f"2. Post-Prefill Token    : shape={next_token.shape} (selected with keepdims=True)")

  # 3. Compiled Single-Token Decode Step with Static Cache Shapes
  compile_counter = {"count": 0}

  @jax.jit
  def compiled_decode_step(token_id, k_buf, v_buf, cache_pos):
    compile_counter["count"] += 1
    x_step = embed_table[token_id]  # [B, 1, D]
    q = (x_step @ w_q).reshape(BATCH_SIZE, 1, NUM_Q_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k_new = (x_step @ w_k).reshape(BATCH_SIZE, 1, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v_new = (x_step @ w_v).reshape(BATCH_SIZE, 1, NUM_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    # In-place static slice update at cache_pos (zero shape change!)
    k_buf = jax.lax.dynamic_update_slice(k_buf, k_new, (0, 0, cache_pos, 0))
    v_buf = jax.lax.dynamic_update_slice(v_buf, v_new, (0, 0, cache_pos, 0))

    # GQA head broadcast: repeat 2 KV heads -> 4 Q heads
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

  # Warmup 2 steps, then rewind StaticCache in-place (exact perflab/steps/decode.py pattern)
  tok_w, k_cache, v_cache = compiled_decode_step(next_token, k_cache, v_cache, jnp.int32(PROMPT_LEN))
  tok_w, k_cache, v_cache = compiled_decode_step(tok_w, k_cache, v_cache, jnp.int32(PROMPT_LEN + 1))
  jax.block_until_ready(tok_w)
  compiles_after_warmup = compile_counter["count"]

  # Rewind cache slice in-place without changing buffer addresses
  k_cache = k_cache.at[:, :, PROMPT_LEN:, :].set(0.0)
  v_cache = v_cache.at[:, :, PROMPT_LEN:, :].set(0.0)

  t0 = time.perf_counter()
  cur_tok = next_token
  generated = []
  for step_idx in range(GENERATE_TOKENS):
    cur_pos = jnp.int32(PROMPT_LEN + step_idx)
    cur_tok, k_cache, v_cache = compiled_decode_step(cur_tok, k_cache, v_cache, cur_pos)
    generated.append(cur_tok)
  jax.block_until_ready(cur_tok)
  decode_ms = (time.perf_counter() - t0) * 1000.0

  compiles_after_decode = compile_counter["count"]
  print(f"3. XLA Compilations During Warmup        : {compiles_after_warmup}")
  print(f"4. New Compilations Across {GENERATE_TOKENS} Decode Steps : {compiles_after_decode - compiles_after_warmup} (ZERO recompilations!)")
  print(f"5. Measured Decode Time ({GENERATE_TOKENS} steps)        : {decode_ms:.3f} ms ({decode_ms / GENERATE_TOKENS:.3f} ms/token)\n")


if __name__ == "__main__":
  run_static_cache_decode_lab()
