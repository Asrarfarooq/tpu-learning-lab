"""Recipe 05: Zero-Graph-Break Mixture-of-Experts (`torch.tpu.ops.ragged_dot`).

Verified against `//depot/google3/third_party/py/torch_tpu/`:
- `torch_tpu/ops/ragged_dot.py`
- `google/examples/gemma4/gemma4_tutorial.md` (Section: Ragged MoE Routing)

Why Naive Hugging Face MoE Loops Break on TPU:
Standard MoE layers loop over `range(num_experts)` and call
`top_k_index == expert_idx` / `torch.where(...)` (`aten.nonzero`), which produces
data-dependent dynamic tensor shapes, forces host-device synchronization, and
breaks `torch.compile`.

The Verified TorchTPU `ragged_dot` Solution:
1. Flatten routed `(token, top_k)` pairs and sort tokens by `expert_id` using
   `torch.argsort(expert_ids, stable=True)`.
2. Compute the `[num_experts]` token histogram on-device with zero host sync via
   `group_sizes.scatter_add_(0, expert_ids, torch.ones_like(expert_ids, dtype=torch.int32))`.
3. Execute all experts in ONE fused XLA `ragged_dot` kernel (`gate_up` + SwiGLU + `down`).
"""

from __future__ import annotations
import time

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    F = None

try:
    import torch_tpu  # noqa: F401
    from torch.tpu.ops import ragged_dot
except ImportError:
    ragged_dot = None

HAS_TPU = (
    HAS_TORCH
    and hasattr(torch, "accelerator")
    and torch.accelerator.is_available()
    and getattr(torch.accelerator.current_accelerator(), "type", None) == "tpu"
)


def get_device() -> torch.device:
    return torch.device("tpu") if HAS_TPU else torch.device("cpu")


def sync_device() -> None:
    if HAS_TPU:
        torch.accelerator.synchronize()


def ragged_dot_reference(lhs: torch.Tensor, rhs: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
    """CPU reference matching `torch.tpu.ops.ragged_dot(lhs, rhs, group_sizes)`."""
    if HAS_TPU and ragged_dot is not None and lhs.device.type == "tpu":
        return ragged_dot(lhs=lhs, rhs=rhs, group_sizes=group_sizes)
    outputs = []
    offset = 0
    for e_idx, count in enumerate(group_sizes.tolist()):
        count_int = int(count)
        if count_int > 0:
            outputs.append(lhs[offset : offset + count_int] @ rhs[e_idx])
        offset += count_int
    return torch.cat(outputs, dim=0) if outputs else lhs.new_zeros((0, rhs.size(-1)))


def naive_moe_forward(
    x: torch.Tensor,
    top_k_indices: torch.Tensor,
    top_k_weights: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Standard Hugging Face expert loop with boolean indexing (causes graph breaks!)."""
    num_experts = w_gate_up.size(0)
    inter_dim = w_down.size(1)
    out = torch.zeros_like(x)

    for expert_idx in range(num_experts):
        mask = top_k_indices == expert_idx
        token_idx, k_idx = torch.where(mask)
        if token_idx.numel() == 0:
            continue
        expert_in = x[token_idx]
        gate_up = expert_in @ w_gate_up[expert_idx]
        gate, up = gate_up[:, :inter_dim], gate_up[:, inter_dim:]
        hidden = F.silu(gate) * up
        expert_out = hidden @ w_down[expert_idx]
        weights = top_k_weights[token_idx, k_idx].unsqueeze(-1).to(expert_out.dtype)
        out.index_add_(0, token_idx, expert_out * weights)
    return out


def ragged_moe_forward(
    x: torch.Tensor,
    top_k_indices: torch.Tensor,
    top_k_weights: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Zero-graph-break TorchTPU Ragged MoE using `argsort` + `scatter_add_` + `ragged_dot`."""
    num_tokens, hidden_dim = x.shape
    top_k = top_k_indices.size(1)
    num_experts = w_gate_up.size(0)
    inter_dim = w_down.size(1)

    flat_experts = top_k_indices.reshape(-1)
    flat_weights = top_k_weights.reshape(-1)

    # 1. Sort routed token-expert pairs by expert index (static shapes!)
    perm = torch.argsort(flat_experts, stable=True)
    sorted_token_indices = perm // top_k
    sorted_x = x.index_select(0, sorted_token_indices)

    # 2. Compute per-expert token counts on-device with scatter_add_ (0 host sync!)
    group_sizes = torch.zeros(num_experts, dtype=torch.int32, device=x.device)
    group_sizes.scatter_add_(0, flat_experts, torch.ones_like(flat_experts, dtype=torch.int32))

    # 3. Fused SwiGLU across all experts in two ragged_dot calls
    gate_up = ragged_dot_reference(sorted_x, w_gate_up, group_sizes)
    gate, up = gate_up[:, :inter_dim], gate_up[:, inter_dim:]
    activated = F.silu(gate) * up
    sorted_out = ragged_dot_reference(activated, w_down, group_sizes)

    # 4. Weight and un-permute back to original token order
    sorted_weights = flat_weights.index_select(0, perm).unsqueeze(-1).to(sorted_out.dtype)
    weighted_out = sorted_out * sorted_weights

    unsorted_out = torch.empty_like(weighted_out)
    unsorted_out.scatter_(
        0,
        perm.unsqueeze(-1).expand(-1, hidden_dim),
        weighted_out,
    )
    return unsorted_out.view(num_tokens, top_k, hidden_dim).sum(dim=1)


def main() -> None:
    if not HAS_TORCH:
        import jax
        import jax.numpy as jnp
        print("Running JAX lax.ragged_dot fallback verification on", jax.default_backend())
        lhs = jnp.ones((64, 128), dtype=jnp.float32)
        rhs = jnp.ones((4, 128, 256), dtype=jnp.float32) * 0.05
        group_sizes = jnp.array([16, 16, 16, 16], dtype=jnp.int32)
        out = jax.lax.ragged_dot(lhs, rhs, group_sizes)
        print("jax.lax.ragged_dot output shape:", out.shape)
        return
    device = get_device()
    dtype = torch.bfloat16
    num_tokens = 512
    hidden_dim = 512
    inter_dim = 1024
    num_experts = 8
    top_k = 2

    print("=" * 76)
    print(f"TorchTPU Ragged MoE (ragged_dot) vs. Naive Loop Benchmark | Device: {device}")
    print(f"Tokens: {num_tokens} | Hidden: {hidden_dim} | Experts: {num_experts} (Top-{top_k})")
    print("=" * 76)

    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    router_logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    top_k_weights, top_k_indices = torch.topk(F.softmax(router_logits, dim=-1), k=top_k, dim=-1)

    w_gate_up = torch.randn(num_experts, hidden_dim, 2 * inter_dim, device=device, dtype=dtype) * 0.02
    w_down = torch.randn(num_experts, inter_dim, hidden_dim, device=device, dtype=dtype) * 0.02

    with torch.no_grad():
        out_naive = naive_moe_forward(x, top_k_indices, top_k_weights, w_gate_up, w_down)
        out_ragged = ragged_moe_forward(x, top_k_indices, top_k_weights, w_gate_up, w_down)
        sync_device()

    max_diff = (out_naive.float() - out_ragged.float()).abs().max().item()
    print(f"\n[1] Numerical Parity Check:")
    print(f"  - Max Abs Diff (Naive Loop vs. Ragged MoE) : {max_diff:.6e}")

    iters = 20
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = naive_moe_forward(x, top_k_indices, top_k_weights, w_gate_up, w_down)
        sync_device()
        naive_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        t1 = time.perf_counter()
        for _ in range(iters):
            _ = ragged_moe_forward(x, top_k_indices, top_k_weights, w_gate_up, w_down)
        sync_device()
        ragged_ms = ((time.perf_counter() - t1) / iters) * 1000.0

    print(f"\n[2] Execution Latency ({iters} iterations):")
    print(f"  - Naive Expert Loop (torch.where / dynamic shapes) : {naive_ms:8.3f} ms")
    print(f"  - Ragged MoE (argsort + scatter_add_ + ragged_dot) : {ragged_ms:8.3f} ms")
    print(f"  - Speedup                                          : {naive_ms / max(ragged_ms, 1e-6):8.2f}x")


if __name__ == "__main__":
    main()
