#!/usr/bin/env python3
'''Module 6, Lab 3: Cloud TPU v5p vs. v6e (Trillium) Workload Sizing & Recipe Advisor.'''

# Verified hardware constants from //depot/google3/third_party/py/torch_tpu/_internal/utils/hardware.py
TPU_SPECS = {
    "v5e": {"hbm_gib": 16.0, "bf16_tflops": 197.0, "hbm_bw_gbs": 819.0, "mxu": "128x128 (4/chip)"},
    "v5p": {"hbm_gib": 95.0, "bf16_tflops": 459.0, "hbm_bw_gbs": 2765.0, "mxu": "128x128 (8/chip)"},
    "v6e": {"hbm_gib": 32.0, "bf16_tflops": 918.0, "hbm_bw_gbs": 1600.0, "mxu": "256x256 (2/chip)"},
    "v7":  {"hbm_gib": 192.0, "bf16_tflops": 2307.0, "hbm_bw_gbs": 7370.0, "mxu": "256x256 (Dual-Core)"},
}

WORKLOADS = [
    {"name": "Qwen/Qwen3-0.6B (LoRA SFT, B=8, S=2048)", "params_b": 0.6, "mode": "lora", "batch": 8, "seq_len": 2048},
    {"name": "meta-llama/Llama-3.2-1B (Full SFT, B=4, S=2048)", "params_b": 1.24, "mode": "full_sft", "batch": 4, "seq_len": 2048},
    {"name": "google/gemma-2-2b (Full SFT, B=4, S=2048)", "params_b": 2.6, "mode": "full_sft", "batch": 4, "seq_len": 2048},
    {"name": "meta-llama/Llama-3.1-8B (LoRA SFT, B=4, S=4096)", "params_b": 8.0, "mode": "lora", "batch": 4, "seq_len": 4096},
    {"name": "meta-llama/Llama-3.1-8B (Full SFT, B=4, S=4096)", "params_b": 8.0, "mode": "full_sft", "batch": 4, "seq_len": 4096},
    {"name": "meta-llama/Llama-3.1-8B (Decode Batch=1, S=2048)", "params_b": 8.0, "mode": "decode", "batch": 1, "seq_len": 2048},
]


def analyze_workloads():
  print("=" * 98)
  print("CLOUD TPU CONSULTING SIZING ADVISOR: VERIFIED HARDWARE SPECS (torch_tpu/_internal/utils/hardware.py)")
  print("=" * 98)
  print(f"{'TPU Gen':<10} | {'HBM / Chip':<12} | {'BF16 Peak / Chip':<18} | {'HBM BW / Chip':<15} | {'Ridge Point (FLOP/B)':<20} | {'MXU Array'}")
  print("-" * 98)
  for gen, s in TPU_SPECS.items():
    ridge = (s["bf16_tflops"] * 1000.0) / s["hbm_bw_gbs"]
    print(f"{gen:<10} | {s['hbm_gib']:>6.0f} GiB   | {s['bf16_tflops']:>10.0f} TFLOP/s | {s['hbm_bw_gbs']:>9.0f} GB/s  | {ridge:>14.1f} FLOP/B | {s['mxu']}")

  print("\n" + "=" * 98)
  print("WORKLOAD MEMORY & PERFORMANCE SIZING: TPU v6e (TRILLIUM) vs. TPU v5p")
  print("=" * 98)
  print(f"{'Workload':<48} | {'Est. HBM':<10} | {'Min v6e Topo':<12} | {'v6e vs v5p Winner & Rationale'}")
  print("-" * 98)

  for w in WORKLOADS:
    p = w["params_b"]
    tokens = w["batch"] * w["seq_len"]
    if w["mode"] == "lora":
      # Frozen bf16 weights (2B/param) + LoRA states (~0.4B/param) + activations
      hbm_gib = (p * 2.4) + (tokens * 0.00035 * p)
      winner = "v6e (2.0x MXU TFLOP/s -> ~1.8x faster step)"
    elif w["mode"] == "full_sft":
      # bf16 weights (2B) + bf16 grads (2B) + AdamW states (8B in bf16 / 12B in fp32) + activations
      hbm_gib = (p * 12.0) + (tokens * 0.0005 * p)
      winner = "v6e (Compute-bound 6*P*T FLOPs per step)"
    else:
      # Decode: bf16 weights (2B/param) + KV cache
      hbm_gib = (p * 2.0) + (tokens * 0.00015 * p)
      winner = "v5p (1.73x HBM bandwidth for B=1 decode)"

    if hbm_gib <= 28.0:
      v6e_topo = "v6e-1 (1 chip)"
    elif hbm_gib <= 112.0:
      v6e_topo = "v6e-4 (4 chips)"
    else:
      v6e_topo = "v6e-8 (8 chips)"

    print(f"{w['name']:<48} | {hbm_gib:>6.1f} GiB | {v6e_topo:<12} | {winner}")


if __name__ == "__main__":
  analyze_workloads()
