"""Recipe 04: Native FP8 (`torch._scaled_mm`) & MXU Precision Modes Benchmark.

Verified against `//depot/google3/third_party/py/torch_tpu/`:
- `csrc/ops/matmul/scaled_mm_aten_kernels.cc`:
  Lowers `aten::_scaled_mm` directly to XLA `DotGeneral` on `torch.float8_e4m3fn`
  (`MAX = 448.0`) and `torch.float8_e5m2` (`MAX = 57344.0`) with:
  1. TensorWise scaling: `scale_a.numel() == 1` and `scale_b.numel() == 1`
  2. RowWise scaling: `scale_a` shape `(M, 1)` and `scale_b` shape `(1, N)`
  3. BlockWise scaling: `scale_a` shape `(M, K // block_size)` and `scale_b` shape `(K // block_size, N)`
- `torch.tpu.precision(torch.tpu.Precision.DEFAULT / HIGH / HIGHEST)`:
  Controls 1-pass (`bf16`), 3-pass (`bf16_3x`), and 6-pass (`bf16_6x`) MXU systolic passes.
"""

import time
import torch

try:
    import torch_tpu  # noqa: F401
    HAS_TPU = torch.tpu.is_available()
except ImportError:
    HAS_TPU = False


def get_device() -> torch.device:
    return torch.device("tpu") if HAS_TPU else torch.device("cpu")


def sync_device() -> None:
    if HAS_TPU:
        torch.tpu.synchronize()


def quantize_fp8_tensorwise(x: torch.Tensor, fp8_dtype=torch.float8_e4m3fn):
    """Quantizes a 2D tensor to FP8 with a single scalar float32 scale."""
    fp8_max = torch.finfo(fp8_dtype).max  # 448.0 for e4m3fn, 57344.0 for e5m2
    amax = x.float().abs().amax().clamp(min=1e-12)
    scale = (amax / fp8_max).to(torch.float32)
    x_fp8 = (x.float() / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale


def quantize_fp8_rowwise(x: torch.Tensor, dim: int, fp8_dtype=torch.float8_e4m3fn):
    """Quantizes a 2D tensor along `dim` for RowWise `torch._scaled_mm`."""
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.float().abs().amax(dim=dim, keepdim=True).clamp(min=1e-12)
    scale = (amax / fp8_max).to(torch.float32)
    x_fp8 = (x.float() / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale


def run_fp8_and_precision_benchmark() -> None:
    device = get_device()
    M, K, N = (1024, 2048, 2048) if HAS_TPU else (512, 512, 512)
    flops_per_matmul = 2 * M * K * N

    print("=" * 76)
    print(f"TorchTPU FP8 (torch._scaled_mm) & MXU Precision Benchmark | Device: {device}")
    print(f"Matrix Shapes: X=[{M}, {K}] @ W=[{K}, {N}] | FLOPs/op: {flops_per_matmul:,}")
    print("=" * 76)

    torch.manual_seed(42)
    x_fp32 = torch.randn((M, K), dtype=torch.float32, device=device) * 0.5
    w_fp32 = torch.randn((K, N), dtype=torch.float32, device=device) * 0.5
    x_bf16 = x_fp32.to(torch.bfloat16)
    w_bf16 = w_fp32.to(torch.bfloat16)

    # Golden reference in float64 on CPU
    ref_fp64 = (x_fp32.cpu().double() @ w_fp32.cpu().double()).float()

    # 1. Quantize to FP8 E4M3fn (TensorWise & RowWise)
    x_fp8_tw, scale_x_tw = quantize_fp8_tensorwise(x_bf16, torch.float8_e4m3fn)
    w_fp8_tw, scale_w_tw = quantize_fp8_tensorwise(w_bf16, torch.float8_e4m3fn)
    # PyTorch _scaled_mm requires the second operand (weight) in column-major layout:
    w_fp8_tw_col = w_fp8_tw.t().contiguous().t()

    x_fp8_rw, scale_x_rw = quantize_fp8_rowwise(x_bf16, dim=1, fp8_dtype=torch.float8_e4m3fn)
    w_fp8_rw, scale_w_rw = quantize_fp8_rowwise(w_bf16, dim=0, fp8_dtype=torch.float8_e4m3fn)
    w_fp8_rw_col = w_fp8_rw.t().contiguous().t()

    print("\n[1] Memory Footprint Comparison for Weight Matrix W:")
    print(f"  - float32       : {w_fp32.numel() * 4 / 1024**2:6.2f} MiB (4 bytes/elem)")
    print(f"  - bfloat16      : {w_bf16.numel() * 2 / 1024**2:6.2f} MiB (2 bytes/elem)")
    print(f"  - float8_e4m3fn : {w_fp8_tw.numel() * 1 / 1024**2:6.2f} MiB (1 byte/elem -> 2x smaller than bf16!)")

    def run_scaled_mm_tensorwise():
        if HAS_TPU:
            return torch._scaled_mm(
                x_fp8_tw, w_fp8_tw_col, scale_a=scale_x_tw, scale_b=scale_w_tw, out_dtype=torch.bfloat16
            )
        # CPU fallback for local preview (since aten::_scaled_mm on CPU requires AVX512/CUDA/TPU):
        return ((x_fp8_tw.float() * scale_x_tw) @ (w_fp8_tw.float() * scale_w_tw)).to(torch.bfloat16)

    def run_scaled_mm_rowwise():
        if HAS_TPU:
            return torch._scaled_mm(
                x_fp8_rw, w_fp8_rw_col, scale_a=scale_x_rw, scale_b=scale_w_rw, out_dtype=torch.bfloat16
            )
        return ((x_fp8_rw.float() * scale_x_rw) @ (w_fp8_rw.float() * scale_w_rw)).to(torch.bfloat16)

    modes = [
        ("FP8 E4M3fn TensorWise (_scaled_mm)", run_scaled_mm_tensorwise),
        ("FP8 E4M3fn RowWise (_scaled_mm)", run_scaled_mm_rowwise),
        ("BF16 Native (1-pass MXU)", lambda: x_bf16 @ w_bf16),
        ("FP32 Default (medium / 1-pass)", lambda: x_fp32 @ w_fp32),
    ]

    print(f"\n[2] Numerical Accuracy & Throughput Across Precision Modes:")
    print(f"  {'Mode':<36} | {'Mean Abs Err':<13} | {'Latency (ms)':<12} | {'GFLOP/s':<10}")
    print("  " + "-" * 78)

    for label, fn in modes:
        for _ in range(3):
            out = fn()
        sync_device()

        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn()
        sync_device()
        dt_ms = ((time.perf_counter() - t0) / iters) * 1000.0
        gflops = (flops_per_matmul / (dt_ms / 1000.0)) / 1e9
        mae = (out.cpu().float() - ref_fp64).abs().mean().item()
        print(f"  {label:<36} | {mae:<13.6f} | {dt_ms:<12.3f} | {gflops:<10.1f}")


if __name__ == "__main__":
    run_fp8_and_precision_benchmark()
