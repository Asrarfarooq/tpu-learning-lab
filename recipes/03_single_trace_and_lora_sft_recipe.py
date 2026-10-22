#!/usr/bin/env python3
'''Recipe 3: Zero-Graph-Break SingleTraceTrainer (torch.tpu.optimizers) & PEFT LoRA Fine-Tuning Template.'''

try:
  import torch
  import torch.nn as nn
  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False


def chunked_unembed_cross_entropy(hidden_states, lm_head_weight, labels, chunk_size: int = 128):
  '''Computes cross-entropy loss in fixed token chunks so full [B, S, V] logits are never materialized.'''
  flat_h = hidden_states.reshape(-1, hidden_states.size(-1))
  flat_labels = labels.reshape(-1)
  num_chunks = flat_h.size(0) // chunk_size
  h_chunks = flat_h.view(num_chunks, chunk_size, -1)
  l_chunks = flat_labels.view(num_chunks, chunk_size)

  total_loss = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
  for i in range(num_chunks):
    logits = (h_chunks[i] @ lm_head_weight.t()).float()
    total_loss = total_loss + nn.functional.cross_entropy(logits, l_chunks[i])
  return total_loss / num_chunks


def main():
  if not HAS_TORCH:
    print("PyTorch not installed in local CPU preview env; switch Execution Target to 'Cloud TPU VM' to run SingleTraceTrainer (torch.tpu.optimizers).")
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

  print(f"Running SingleTrace / Fused AdamW Fine-Tuning Recipe on device={device}...")
  model = nn.Sequential(
      nn.Linear(256, 512, dtype=torch.bfloat16),
      nn.SiLU(),
      nn.Linear(512, 256, dtype=torch.bfloat16),
  ).to(device)

  x = torch.randn(4, 128, 256, dtype=torch.bfloat16, device=device)
  target = torch.randn(4, 128, 256, dtype=torch.bfloat16, device=device)

  # Use public torch.tpu.optimizers.SingleTraceTrainer on TorchTPU for 0 graph breaks + buffer donation
  if device.type == "tpu":
    try:
      from torch.tpu import optimizers
      trainer = optimizers.SingleTraceTrainer(
          model=model,
          optimizer=optimizers.AdamW(lr=1e-3, weight_decay=0.01, use_bfloat16_moments=True),
          compile_fn=lambda gm, example_inputs: torch.compile(gm, backend="tpu", fullgraph=True),
      )
      train_step = trainer.make_compiled_train_step()
      for step in range(1, 11):
        loss = train_step(x, target)
      torch.accelerator.synchronize()
      print(f"SingleTraceTrainer (torch.tpu.optimizers.AdamW) Step 10 Loss: {loss.item():.6f}")
      return
    except ImportError:
      pass

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  for step in range(1, 11):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
  print(f"Standard Step 10 Loss: {loss.item():.6f}")


if __name__ == "__main__":
  main()
