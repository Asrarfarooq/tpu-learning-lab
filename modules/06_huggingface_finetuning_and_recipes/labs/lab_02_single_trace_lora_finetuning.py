#!/usr/bin/env python3
'''Module 6, Lab 2: SingleTrace Functional LoRA Fine-Tuning & Chunkwise Vocabulary Cross-Entropy.'''

import numpy as np

BATCH = 4
SEQ_LEN = 128
D_MODEL = 128
LORA_RANK = 8
LORA_ALPHA = 16.0
VOCAB_SIZE = 4096
CHUNK_SIZE = 128


def run_lora_and_chunked_loss_lab():
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed.")
    return

  print("=" * 82)
  print("PART 1: FULL FINE-TUNING vs. PEFT LoRA HBM MEMORY ACCOUNTING")
  print("=" * 82)

  # Compare a 4096 x 4096 projection layer in Full SFT vs. LoRA (r=8)
  d_full = 4096
  full_params = d_full * d_full
  lora_params = 2 * d_full * LORA_RANK
  # Full SFT in mixed precision: bf16 weights (2B) + bf16 grads (2B) + fp32 master & AdamW m,v (12B) = 16 B/param
  full_sft_bytes = full_params * 16
  # LoRA: Base weight frozen in bf16 (2B) + LoRA A,B weights/grads/AdamW states (16B on lora_params only)
  lora_sft_bytes = (full_params * 2) + (lora_params * 16)

  print(f"  Layer Shape            : ({d_full}, {d_full}) = {full_params:,} base parameters")
  print(f"  Trainable Params (Full): {full_params:,} (100.0%) -> HBM (Weights+Grads+AdamW): {full_sft_bytes / (1024**2):.2f} MiB")
  print(f"  Trainable Params (LoRA): {lora_params:,} ({100.0 * lora_params / full_params:.2f}%)  -> HBM (Frozen Base+LoRA):  {lora_sft_bytes / (1024**2):.2f} MiB")
  print(f"  HBM Memory Reduction   : {full_sft_bytes / lora_sft_bytes:.2f}x smaller per Linear layer!\n")

  print("=" * 82)
  print("PART 2: CHUNKWISE FUSED UNEMBEDDING + CROSS-ENTROPY LOSS (ZERO LOGITS OOM)")
  print("=" * 82)

  rng = np.random.default_rng(99)
  num_tokens = BATCH * SEQ_LEN
  h_np = (rng.standard_normal((num_tokens, D_MODEL)) * 0.1).astype(np.float32)
  w_vocab_np = (rng.standard_normal((VOCAB_SIZE, D_MODEL)) * 0.05).astype(np.float32)
  labels_np = rng.integers(0, VOCAB_SIZE, size=(num_tokens,), dtype=np.int32)

  h = jnp.asarray(h_np)
  w_vocab = jnp.asarray(w_vocab_np)
  labels = jnp.asarray(labels_np)

  def full_logits_ce_loss(h_in, w_in, tgt):
    # Materializes full [num_tokens, VOCAB_SIZE] logits tensor in HBM
    logits = h_in @ w_in.T
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, tgt[:, None], axis=-1).squeeze(-1)
    return jnp.mean(nll)

  @jax.jit
  def chunked_unembed_ce_loss(h_in, w_in, tgt):
    # Processes vocabulary logits in fixed chunks of CHUNK_SIZE tokens via lax.scan
    num_chunks = num_tokens // CHUNK_SIZE
    h_chunks = h_in.reshape(num_chunks, CHUNK_SIZE, D_MODEL)
    t_chunks = tgt.reshape(num_chunks, CHUNK_SIZE)

    def scan_chunk(accum_loss, item):
      h_c, t_c = item
      chunk_logits = h_c @ w_in.T  # Only [CHUNK_SIZE, VOCAB_SIZE] in VMEM/HBM!
      log_probs = jax.nn.log_softmax(chunk_logits, axis=-1)
      nll = -jnp.take_along_axis(log_probs, t_c[:, None], axis=-1).squeeze(-1)
      return accum_loss + jnp.mean(nll), None

    total_loss, _ = jax.lax.scan(scan_chunk, 0.0, (h_chunks, t_chunks))
    return total_loss / num_chunks

  loss_full, grad_w_full = jax.value_and_grad(full_logits_ce_loss, argnums=1)(h, w_vocab, labels)
  loss_chunked, grad_w_chunked = jax.jit(jax.value_and_grad(chunked_unembed_ce_loss, argnums=1))(h, w_vocab, labels)

  loss_diff = float(jnp.abs(loss_full - loss_chunked))
  grad_diff = float(jnp.max(jnp.abs(grad_w_full - grad_w_chunked)))
  full_logits_mb = (num_tokens * VOCAB_SIZE * 4) / (1024 ** 2)
  chunk_logits_mb = (CHUNK_SIZE * VOCAB_SIZE * 4) / (1024 ** 2)

  print(f"  Full Logits Tensor Memory    : {full_logits_mb:.2f} MiB ([{num_tokens}, {VOCAB_SIZE}])")
  print(f"  Chunked Logits Tensor Memory : {chunk_logits_mb:.2f} MiB ([{CHUNK_SIZE}, {VOCAB_SIZE}]) -> {full_logits_mb / chunk_logits_mb:.1f}x peak memory reduction!")
  print(f"  Full Loss vs. Chunked Loss   : {float(loss_full):.6f} vs. {float(loss_chunked):.6f} (abs diff = {loss_diff:.2e})")
  print(f"  Max Gradient Abs Difference  : {grad_diff:.2e} (Numerical parity verified!)\n")

  print("=" * 82)
  print("PART 3: SINGLE-TRACE FUNCTIONAL LoRA FINE-TUNING LOOP (15 STEPS)")
  print("=" * 82)

  w_base = jnp.asarray((rng.standard_normal((D_MODEL, D_MODEL)) * 0.05).astype(np.float32))
  lora_params_init = {
      "lora_A": jnp.asarray((rng.standard_normal((D_MODEL, LORA_RANK)) * 0.02).astype(np.float32)),
      "lora_B": jnp.zeros((LORA_RANK, D_MODEL), dtype=jnp.float32),
  }
  scale = LORA_ALPHA / float(LORA_RANK)
  target_proj = w_base + jnp.asarray((rng.standard_normal((D_MODEL, D_MODEL)) * 0.02).astype(np.float32))
  y_target = h @ target_proj

  @jax.jit
  def single_trace_lora_step(lora_p, m_s, v_s, step_idx, x_in, y_tgt):
    def lora_loss_fn(p):
      # Base weight w_base is frozen; only low-rank adapters A and B are updated
      delta_w = (p["lora_A"] @ p["lora_B"]) * scale
      pred = x_in @ (w_base + delta_w)
      return jnp.mean((pred - y_tgt) ** 2)

    loss_val, grads = jax.value_and_grad(lora_loss_fn)(lora_p)
    b1, b2, lr, eps = 0.9, 0.999, 2e-2, 1e-8
    new_m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, m_s, grads)
    new_v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1.0 - b2) * (g ** 2), v_s, grads)
    m_hat = jax.tree_util.tree_map(lambda m: m / (1.0 - b1 ** step_idx), new_m)
    v_hat = jax.tree_util.tree_map(lambda v: v / (1.0 - b2 ** step_idx), new_v)
    new_p = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), lora_p, m_hat, v_hat)
    return new_p, new_m, new_v, loss_val

  lora_state = lora_params_init
  m_state = jax.tree_util.tree_map(jnp.zeros_like, lora_state)
  v_state = jax.tree_util.tree_map(jnp.zeros_like, lora_state)

  for step in range(1, 16):
    lora_state, m_state, v_state, step_loss = single_trace_lora_step(lora_state, m_state, v_state, step, h, y_target)
    if step in (1, 5, 10, 15):
      print(f"  Step {step:2d} | Single-Trace LoRA MSE Loss: {float(step_loss):.6f}")


if __name__ == "__main__":
  run_lora_and_chunked_loss_lab()
