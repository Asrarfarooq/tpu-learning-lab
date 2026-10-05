#!/usr/bin/env python3
'''Module 4, Lab 2: Calling Custom JAX/Pallas Kernels Inside PyTorch via torch.tpu.pallas.jax_op.'''

import numpy as np


def run_torchtpu_pallas_bridge():
  try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
  except ImportError:
    print("[JAX / Pallas] Required libraries not installed.")
    return

  def _pallas_rmsnorm_kernel(x_ref, w_ref, o_ref):
    tile = x_ref[:, :]
    w_tile = w_ref[:, :]
    rms = jax.lax.rsqrt(jnp.mean(jnp.square(tile), axis=-1, keepdims=True) + 1e-6)
    o_ref[:, :] = tile * rms * w_tile

  def jax_pallas_rmsnorm(x, w):
    m, n = x.shape
    bm = min(m, 128)
    w_2d = jnp.broadcast_to(w.reshape(1, n), (m, n))
    return pl.pallas_call(
        _pallas_rmsnorm_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(m // bm, 1),
        in_specs=[
            pl.BlockSpec(block_shape=(bm, n), index_map=lambda i, j: (i, j)),
            pl.BlockSpec(block_shape=(bm, n), index_map=lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec(block_shape=(bm, n), index_map=lambda i, j: (i, j)),
        interpret=(jax.default_backend() == "cpu"),
    )(x, w_2d)

  rng = np.random.default_rng(0)
  x_np = rng.standard_normal((256, 128), dtype=np.float32)
  w_np = np.ones((128,), dtype=np.float32)
  ref_np = x_np * (1.0 / np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + 1e-6)) * w_np

  out_jax = np.asarray(jax_pallas_rmsnorm(jnp.asarray(x_np), jnp.asarray(w_np)))
  jax_diff = float(np.max(np.abs(out_jax - ref_np)))
  print("=" * 78)
  print(f"PALLAS RMSNORM KERNEL & TORCHTPU BRIDGE (JAX backend={jax.default_backend()})")
  print("=" * 78)
  print(f"  JAX Pallas Kernel Output Shape : {out_jax.shape}")
  print(f"  JAX Pallas Kernel Max Error    : {jax_diff:.2e}")

  try:
    import torch
    has_tpu = False
    try:
      import torch_tpu  # noqa: F401
      has_tpu = hasattr(torch, "tpu") and torch.tpu.is_available()
    except ImportError:
      pass
    device = torch.device("tpu:0" if has_tpu else "cpu")

    if has_tpu and hasattr(torch.tpu, "pallas"):
      pallas_rmsnorm_op = torch.tpu.pallas.jax_op("tpu_lab::pallas_rmsnorm", jax_pallas_rmsnorm)
    else:
      @torch.library.custom_op("tpu_lab::pallas_rmsnorm", mutates_args=())
      def pallas_rmsnorm_op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return x * rms * w

      @pallas_rmsnorm_op.register_fake
      def _(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    x_t = torch.from_numpy(x_np).to(device)
    w_t = torch.from_numpy(w_np).to(device)
    out_custom = pallas_rmsnorm_op(x_t, w_t)
    t_diff = float(np.max(np.abs(out_custom.cpu().numpy() - ref_np)))
    print(f"  PyTorch Custom Op ({device}) Max Error: {t_diff:.2e}")
  except ImportError:
    print("  [PyTorch] Not installed in local CPU env; will execute torch.tpu.pallas.jax_op on Cloud TPU VM.")


if __name__ == "__main__":
  run_torchtpu_pallas_bridge()
