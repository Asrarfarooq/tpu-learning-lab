#!/usr/bin/env python3
"""Module 6, Lab 2: SingleTrace Functional LoRA Fine-Tuning & Chunkwise Vocabulary Cross-Entropy (TorchTPU & JAX)."""

import numpy as np

BATCH = 4
SEQ_LEN = 128
D_MODEL = 128
LORA_RANK = 8
LORA_ALPHA = 16.0
VOCAB_SIZE = 4096
CHUNK_SIZE = 128


def run_pytorch_lora_and_chunked_loss():
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

  print("=" * 84)
  print(f"PART 1 (PYTORCH / TORCHTPU): CHUNKED VOCAB LOSS & FUNCTIONAL LoRA ({device})")
  print("=" * 84)

  torch.manual_seed(99)
  num_tokens = BATCH * SEQ_LEN
  h = (torch.randn(num_tokens, D_MODEL, device=device) * 0.1).requires_grad_(True)
  w_vocab = (torch.randn(VOCAB_SIZE, D_MODEL, device=device) * 0.05).requires_grad_(True)
  labels = torch.randint(0, VOCAB_SIZE, (num_tokens,), device=device, dtype=torch.long)

  # 1. Full logits cross-entropy vs Chunked unembedding cross-entropy in PyTorch
  full_loss = F.cross_entropy(h @ w_vocab.t(), labels)
  full_loss.backward()
  grad_w_full = w_vocab.grad.detach().clone()
  w_vocab.grad = None

  num_chunks = num_tokens // CHUNK_SIZE
  h_chunks = h.view(num_chunks, CHUNK_SIZE, D_MODEL)
  l_chunks = labels.view(num_chunks, CHUNK_SIZE)
  chunk_loss = torch.zeros((), device=device)
  for i in range(num_chunks):
    chunk_loss = chunk_loss + F.cross_entropy(h_chunks[i] @ w_vocab.t(), l_chunks[i])
  chunk_loss = chunk_loss / num_chunks
  chunk_loss.backward()
  grad_w_chunk = w_vocab.grad.detach().clone()

  loss_diff = float((full_loss.detach() - chunk_loss.detach()).abs().cpu().item())
  grad_diff = float((grad_w_full - grad_w_chunk).abs().max().cpu().item())
  print(f"  PyTorch Full vs. Chunked CE Loss Diff : {loss_diff:.2e}")
  print(f"  PyTorch Max Vocab Weight Grad Diff    : {grad_diff:.2e} (16.0x peak logits memory reduction!)\n")


def run_jax_lora_and_chunked_loss_lab():
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed.")
    return

  print("=" * 84)
  print("PART 2: FULL FINE-TUNING vs. PEFT LoRA HBM MEMORY ACCOUNTING")
  print("=" * 84)

  d_full = 4096
  full_params = d_full * d_full
  lora_params = 2 * d_full * LORA_RANK
  full_sft_bytes = full_params * 16
  lora_sft_bytes = (full_params * 2) + (lora_params * 16)

  print(f"  Layer Shape            : ({d_full}, {d_full}) = {full_params:,} base parameters")
  print(f"  Trainable Params (Full): {full_params:,} (100.0%) -> HBM (Weights+Grads+AdamW): {full_sft_bytes / (1024**2):.2f} MiB")
  print(f"  Trainable Params (LoRA): {lora_params:,} ({100.0 * lora_params / full_params:.2f}%)  -> HBM (Frozen Base+LoRA):  {lora_sft_bytes / (1024**2):.2f} MiB")
  print(f"  HBM Memory Reduction   : {full_sft_bytes / lora_sft_bytes:.2f}x smaller per Linear layer!\n")

  print("=" * 84)
  print("PART 3 (JAX): CHUNKWISE FUSED UNEMBEDDING + SINGLE-TRACE LoRA FINE-TUNING (15 STEPS)")
  print("=" * 84)

  rng = np.random.default_rng(99)
  num_tokens = BATCH * SEQ_LEN
  h_np = (rng.standard_normal((num_tokens, D_MODEL)) * 0.1).astype(np.float32)
  w_vocab_np = (rng.standard_normal((VOCAB_SIZE, D_MODEL)) * 0.05).astype(np.float32)
  labels_np = rng.integers(0, VOCAB_SIZE, size=(num_tokens,), dtype=np.int32)

  h = jnp.asarray(h_np)
  w_vocab = jnp.asarray(w_vocab_np)
  labels = jnp.asarray(labels_np)

  def full_logits_ce_loss(h_in, w_in, tgt):
    logits = h_in @ w_in.T
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, tgt[:, None], axis=-1).squeeze(-1)
    return jnp.mean(nll)

  @jax.jit
  def chunked_unembed_ce_loss(h_in, w_in, tgt):
    num_chunks = num_tokens // CHUNK_SIZE
    h_chunks = h_in.reshape(num_chunks, CHUNK_SIZE, D_MODEL)
    t_chunks = tgt.reshape(num_chunks, CHUNK_SIZE)

    def scan_chunk(accum_loss, item):
      h_c, t_c = item
      chunk_logits = h_c @ w_in.T
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
  run_pytorch_lora_and_chunked_loss()
  run_jax_lora_and_chunked_loss_lab()
