#!/usr/bin/env python3
'''Module 0, Lab 2: Analytical vs. Autograd vs. JAX value_and_grad Linear Regression.'''

import numpy as np

TRUE_WEIGHTS = [2.5, -1.5, 4.0]
TRUE_BIAS = 0.75
LEARNING_RATE = 0.08
STEPS = 50
NUM_SAMPLES = 256


def generate_dataset():
  rng = np.random.default_rng(0)
  x_np = rng.standard_normal((NUM_SAMPLES, 3), dtype=np.float32)
  w_np = np.array(TRUE_WEIGHTS, dtype=np.float32).reshape(3, 1)
  y_np = x_np @ w_np + TRUE_BIAS
  return x_np, y_np


def verify_step0_gradients(x_np, y_np):
  print("=" * 76)
  print("PART 1: STEP 0 GRADIENT PARITY (CALCULUS vs. PYTORCH AUTOGRAD vs. JAX)")
  print("=" * 76)

  w0_np = np.zeros((3, 1), dtype=np.float32)
  b0_np = np.zeros((1,), dtype=np.float32)

  y_hat = x_np @ w0_np + b0_np
  err = y_hat - y_np
  analytical_dw = (2.0 / NUM_SAMPLES) * (x_np.T @ err)
  analytical_db = (2.0 / NUM_SAMPLES) * np.sum(err, keepdims=True)
  print(f"[Analytical Calculus] dL/dW = {analytical_dw.squeeze().round(5).tolist()} | dL/db = {float(analytical_db.squeeze()):.5f}")

  try:
    import torch
    w_t = torch.zeros((3, 1), dtype=torch.float32, requires_grad=True)
    b_t = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    loss_t = (((torch.from_numpy(x_np) @ w_t + b_t) - torch.from_numpy(y_np)) ** 2).mean()
    loss_t.backward()
    print(f"[PyTorch Autograd]    dL/dW = {w_t.grad.squeeze().numpy().round(5).tolist()} | dL/db = {float(b_t.grad.item()):.5f}")
  except ImportError:
    pass

  try:
    import jax
    import jax.numpy as jnp
    params = {"w": jnp.zeros((3, 1)), "b": jnp.zeros((1,))}
    x_j, y_j = jnp.asarray(x_np), jnp.asarray(y_np)
    _, grads = jax.value_and_grad(lambda p: jnp.mean(((x_j @ p["w"] + p["b"]) - y_j) ** 2))(params)
    print(f"[JAX value_and_grad]  dL/dW = {np.asarray(grads['w']).squeeze().round(5).tolist()} | dL/db = {float(grads['b'][0]):.5f}")
  except ImportError:
    pass
  print()


def train_pytorch(x_np, y_np):
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

  print("=" * 76)
  print(f"PART 2: PYTORCH TRAINING LOOP (device={device})")
  print("=" * 76)

  x = torch.from_numpy(x_np).to(device)
  y_true = torch.from_numpy(y_np).to(device)

  w = torch.zeros((3, 1), dtype=torch.float32, device=device, requires_grad=True)
  b = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)

  for step in range(STEPS + 1):
    y_pred = x @ w + b
    loss = ((y_pred - y_true) ** 2).mean()
    loss.backward()

    if step % 10 == 0:
      print(f"  Step {step:02d} | MSE Loss: {loss.item():.8f}")

    with torch.no_grad():
      w -= LEARNING_RATE * w.grad
      b -= LEARNING_RATE * b.grad

    w.grad = None
    b.grad = None

  learned_w = [round(v, 4) for v in w.detach().cpu().squeeze().tolist()]
  learned_b = round(b.detach().cpu().item(), 4)
  print(f"  Ground Truth : W={TRUE_WEIGHTS}, b={TRUE_BIAS}")
  print(f"  PyTorch Final: W={learned_w}, b={learned_b}\n")


def train_jax(x_np, y_np):
  try:
    import jax
    import jax.numpy as jnp
  except ImportError:
    print("[JAX] Not installed; skipping.")
    return

  print("=" * 76)
  print(f"PART 3: JAX TRAINING LOOP (@jax.jit + value_and_grad on backend={jax.default_backend()})")
  print("=" * 76)

  x = jnp.asarray(x_np)
  y_true = jnp.asarray(y_np)
  params = {"w": jnp.zeros((3, 1), dtype=jnp.float32), "b": jnp.zeros((1,), dtype=jnp.float32)}

  def mse_loss(p, inputs, targets):
    preds = inputs @ p["w"] + p["b"]
    return jnp.mean((preds - targets) ** 2)

  @jax.jit
  def train_step(p, inputs, targets):
    loss_val, grads = jax.value_and_grad(mse_loss)(p, inputs, targets)
    updated_p = jax.tree_util.tree_map(lambda param, g: param - LEARNING_RATE * g, p, grads)
    return updated_p, loss_val

  for step in range(STEPS + 1):
    params, loss_val = train_step(params, x, y_true)
    if step % 10 == 0:
      print(f"  Step {step:02d} | MSE Loss: {float(loss_val):.8f}")

  learned_w = [round(float(v), 4) for v in params["w"].squeeze()]
  learned_b = round(float(params["b"][0]), 4)
  print(f"  Ground Truth : W={TRUE_WEIGHTS}, b={TRUE_BIAS}")
  print(f"  JAX Final    : W={learned_w}, b={learned_b}\n")


if __name__ == "__main__":
  x_data, y_data = generate_dataset()
  verify_step0_gradients(x_data, y_data)
  train_pytorch(x_data, y_data)
  train_jax(x_data, y_data)
