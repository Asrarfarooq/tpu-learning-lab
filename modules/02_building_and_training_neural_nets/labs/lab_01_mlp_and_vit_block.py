#!/usr/bin/env python3
'''Module 2, Lab 1: Modern Transformer Block (RMSNorm + SwiGLU) & AdamW Memory Accounting.'''

import numpy as np

BATCH = 32
SEQ_LEN = 64
D_MODEL = 128
D_FF = 256


def init_shared_weights():
  rng = np.random.default_rng(7)
  return {
      "rms_weight": np.ones((D_MODEL,), dtype=np.float32),
      "w_gate": (rng.standard_normal((D_MODEL, D_FF)) * 0.04).astype(np.float32),
      "w_up": (rng.standard_normal((D_MODEL, D_FF)) * 0.04).astype(np.float32),
      "w_down": (rng.standard_normal((D_FF, D_MODEL)) * 0.04).astype(np.float32),
      "x": rng.standard_normal((BATCH, SEQ_LEN, D_MODEL)).astype(np.float32),
      "target": rng.standard_normal((BATCH, SEQ_LEN, D_MODEL)).astype(np.float32),
  }


def run_pytorch_block(data):
  try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
  except ImportError:
    print("[PyTorch] Not installed; skipping.")
    return None

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

  class RMSNormSwiGLUBlock(nn.Module):
    def __init__(self):
      super().__init__()
      self.rms_weight = nn.Parameter(torch.from_numpy(data["rms_weight"].copy()))
      self.w_gate = nn.Parameter(torch.from_numpy(data["w_gate"].copy()))
      self.w_up = nn.Parameter(torch.from_numpy(data["w_up"].copy()))
      self.w_down = nn.Parameter(torch.from_numpy(data["w_down"].copy()))

    def forward(self, x):
      # Upcast to float32 for variance stability, then restore dtype
      rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype)
      normed = x * rms * self.rms_weight.to(x.dtype)
      gated = F.silu(normed @ self.w_gate) * (normed @ self.w_up)
      return x + (gated @ self.w_down)

  model = RMSNormSwiGLUBlock().to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)

  x = torch.from_numpy(data["x"]).to(device)
  target = torch.from_numpy(data["target"]).to(device)

  out0 = model(x).detach().cpu().numpy()

  for step in range(15):
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = F.mse_loss(pred, target)
    loss.backward()
    optimizer.step()

  param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
  opt_bytes = sum(
      v.numel() * v.element_size()
      for state in optimizer.state.values()
      for v in state.values()
      if torch.is_tensor(v)
  )
  print("=" * 78)
  print(f"PYTORCH RMSNorm + SwiGLU BLOCK (device={device})")
  print("=" * 78)
  print(f"  Parameter Count        : {sum(p.numel() for p in model.parameters()):,}")
  print(f"  Weight Memory (fp32)   : {param_bytes:,} bytes")
  print(f"  AdamW State Memory     : {opt_bytes:,} bytes (2x weight bytes for m_t and v_t!)")
  print(f"  Step 15 MSE Loss       : {loss.item():.6f}\n")
  return out0


def run_jax_block(data, torch_out0):
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed; skipping.")
    return

  params = {
      "rms_weight": jnp.asarray(data["rms_weight"]),
      "w_gate": jnp.asarray(data["w_gate"]),
      "w_up": jnp.asarray(data["w_up"]),
      "w_down": jnp.asarray(data["w_down"]),
  }
  x = jnp.asarray(data["x"])
  target = jnp.asarray(data["target"])

  def forward(p, inp):
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(inp.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
    normed = inp * rms.astype(inp.dtype) * p["rms_weight"]
    gated = jax.nn.silu(normed @ p["w_gate"]) * (normed @ p["w_up"])
    return inp + (gated @ p["w_down"])

  jax_out0 = np.asarray(forward(params, x))
  print("=" * 78)
  print(f"JAX FUNCTIONAL PYTREE RMSNorm + SwiGLU BLOCK (backend={jax.default_backend()})")
  print("=" * 78)
  if torch_out0 is not None:
    max_diff = float(np.max(np.abs(torch_out0 - jax_out0)))
    print(f"  Step 0 Forward Parity (PyTorch vs. JAX Max Abs Diff): {max_diff:.2e}")

  # Explicit AdamW state PyTrees: m (1st moment) and v (2nd moment)
  m_state = jax.tree_util.tree_map(jnp.zeros_like, params)
  v_state = jax.tree_util.tree_map(jnp.zeros_like, params)

  @jax.jit
  def adamw_step(p, m, v, step_num, inp, tgt, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    loss_val, grads = jax.value_and_grad(lambda par: jnp.mean((forward(par, inp) - tgt) ** 2))(p)
    new_m = jax.tree_util.tree_map(lambda m_i, g_i: b1 * m_i + (1.0 - b1) * g_i, m, grads)
    new_v = jax.tree_util.tree_map(lambda v_i, g_i: b2 * v_i + (1.0 - b2) * (g_i ** 2), v, grads)
    m_hat = jax.tree_util.tree_map(lambda m_i: m_i / (1.0 - b1 ** step_num), new_m)
    v_hat = jax.tree_util.tree_map(lambda v_i: v_i / (1.0 - b2 ** step_num), new_v)
    new_p = jax.tree_util.tree_map(
        lambda p_i, mh, vh: p_i - lr * (mh / (jnp.sqrt(vh) + eps) + wd * p_i),
        p, m_hat, v_hat,
    )
    return new_p, new_m, new_v, loss_val

  for step in range(1, 16):
    params, m_state, v_state, loss_val = adamw_step(params, m_state, v_state, step, x, target)

  print(f"  Step 15 MSE Loss       : {float(loss_val):.6f}\n")


if __name__ == "__main__":
  shared = init_shared_weights()
  t_out = run_pytorch_block(shared)
  run_jax_block(shared, t_out)
