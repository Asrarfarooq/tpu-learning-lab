#!/usr/bin/env python3
'''Recipe 2: Hugging Face LLM Prefill + StaticCache Decode Benchmark on TorchTPU (v5p / v6e).'''

import argparse
import json
import time

try:
  import torch
  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False


def main():
  parser = argparse.ArgumentParser(description="TorchTPU Hugging Face LLM Benchmark")
  parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
  parser.add_argument("--batch-size", type=int, default=2)
  parser.add_argument("--prompt-len", type=int, default=512)
  parser.add_argument("--decode-steps", type=int, default=16)
  args = parser.parse_args()

  if not HAS_TORCH:
    print("PyTorch not installed in local CPU preview env; switch Execution Target to 'Cloud TPU VM' to run full Hugging Face Qwen3/Llama-3.2 StaticCache benchmark.")
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
  device = torch.device("tpu" if has_tpu else "cpu")

  import transformers
  from transformers import cache_utils

  print(f"Loading config for {args.model_id} on {device} (bfloat16)...")
  config = transformers.AutoConfig.from_pretrained(args.model_id)
  config.use_cache = True

  with torch.device(device):
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16).eval()
    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.prompt_len), dtype=torch.long)

  cache = cache_utils.StaticCache(
      config=config,
      max_cache_len=args.prompt_len + args.decode_steps,
      batch_size=args.batch_size,
      dtype=torch.bfloat16,
      device=device,
  )

  # 1. Eager Prefill with torch.no_grad() (never @torch.inference_mode() with torch.compile on TPU)
  t0 = time.perf_counter()
  with torch.no_grad(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.OVERRIDEABLE]):
    prefill_out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    next_token = torch.argmax(prefill_out.logits[:, -1, :], dim=-1, keepdim=True)
  if device.type == "tpu":
    torch.accelerator.synchronize()
  prefill_ms = (time.perf_counter() - t0) * 1000.0

  # 2. Compile Decode Step with dynamic=False
  compiled_decode = (
      torch.compile(model, backend="tpu", dynamic=False)
      if device.type == "tpu"
      else model
  )

  # Warmup 2 steps, then rewind StaticCache in-place
  with torch.no_grad():
    for w_idx in range(2):
      pos = torch.tensor([args.prompt_len + w_idx], device=device)
      out = compiled_decode(input_ids=next_token, past_key_values=cache, cache_position=pos, use_cache=True)
      next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
  if device.type == "tpu":
    torch.accelerator.synchronize()

  for layer in getattr(cache, "layers", []):
    if getattr(layer, "keys", None) is not None:
      layer.keys[:, :, args.prompt_len:, :].zero_()
    if getattr(layer, "values", None) is not None:
      layer.values[:, :, args.prompt_len:, :].zero_()

  # 3. Timed Decode Steps
  t1 = time.perf_counter()
  with torch.no_grad():
    for s_idx in range(args.decode_steps):
      pos = torch.tensor([args.prompt_len + s_idx], device=device)
      out = compiled_decode(input_ids=next_token, past_key_values=cache, cache_position=pos, use_cache=True)
      next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
  if device.type == "tpu":
    torch.accelerator.synchronize()
  decode_ms = (time.perf_counter() - t1) * 1000.0

  report = {
      "model_id": args.model_id,
      "device": str(device),
      "batch_size": args.batch_size,
      "prompt_len": args.prompt_len,
      "decode_steps": args.decode_steps,
      "prefill_ttft_ms": round(prefill_ms, 3),
      "decode_itl_ms_per_step": round(decode_ms / args.decode_steps, 3),
      "decode_tokens_per_sec": round((args.batch_size * args.decode_steps) / (decode_ms / 1000.0), 2),
  }
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
