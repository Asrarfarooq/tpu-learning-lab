# Unit 6.2: Zero-Break Training & Fine-Tuning — `SingleTraceTrainer`, PEFT LoRA & Chunked Vocab Loss

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Diagnose why standard PyTorch `loss.backward()` + `optimizer.step()` causes graph breaks under `torch.compile`, and eliminate them using the official public **`torch.tpu.optimizers`** API (**`optimizers.SingleTraceTrainer`**, **`optimizers.AdamW`**, **`optimizers.Muon`**, **`optimizers.Sgd`**) or **`patch_optimizer_graph_breaks()`**.
2. Configure **PEFT LoRA (`peft.LoraConfig`)** for Supervised Fine-Tuning (SFT) on TorchTPU with fixed-length padding (`padding="max_length"`), `lora_dropout=0.0`, and `num_workers=0`.
3. Implement **Chunkwise Fused Unembedding + Cross-Entropy Loss** (`ChunkwiseFusedUnembedCrossEntropyLoss`) to prevent multi-gigabyte `[batch, seq_len, vocab_size]` logits OOMs on large-vocabulary models (`Gemma` 256k vocab, `Qwen3` 152k vocab).

## 2. Why Standard Training Steps Break `torch.compile` (And Two Verified Fixes)

If you wrap a standard PyTorch training function (`output = model(x); loss.backward(); optimizer.step()`) in `torch.compile(fn, backend="tpu", dynamic=False)`, TorchDynamo cannot trace through the C++ autograd engine call (`loss.backward()`) or the `_use_grad_for_differentiable` decorator on `torch.optim.Optimizer.step()`.

You can verify this programmatically using `torch._dynamo.explain`:

```python
explanation = torch._dynamo.explain(basic_train_step)(x, target)
print("Graph breaks:", explanation.graph_break_count)
# -> Splits execution into separate Forward, Backward, and per-parameter Optimizer subgraphs!
```

On Gemma-4 (`google/examples/gemma4/gemma4_tutorial.md`), an uncompiled/broken optimizer step stalls the host CPU for **`712 ms` per step** out of an `882 ms` total step time! TorchTPU provides two verified patterns to fuse **`Forward + Loss + Backward + Optimizer`** into a **single unbroken XLA compilation**:

| Pattern | TorchTPU Public API (`0.2.0+`) | How It Works | When to Use |
| :--- | :--- | :--- | :--- |
| **Pattern A: `SingleTraceTrainer`** | `from torch.tpu import optimizers` (`optimizers.SingleTraceTrainer`, `optimizers.AdamW`, `optimizers.Muon`, `optimizers.Sgd`) | Uses `torch.func.grad_and_value` + functional optimizer (`AdamW` / `Muon` / `Sgd`) + `make_fx` to capture `FWD+BWD+OPT` into 1 graph with `fullgraph=True`, input buffer donation, tied-weights deduplication (`tie_weights=True`), `DTensor` unwrapping (`SimpleFSDP`), and automatic Hugging Face `transformers.masking_utils.is_tracing` patching. | **Gold standard** for maximum single-device and `SimpleFSDP` throughput with `bfloat16` optimizer moments. |
| **Pattern B: Optimizer Graph-Break Patch** | `from torch_tpu._internal.optim.patch import patch_optimizer_graph_breaks` | Strips the graph-breaking decorator from `Optimizer.step` and eagerly pre-initializes `exp_avg` / `exp_avg_sq` on step 0 so `@torch.compile` traces the full training step. | When using standard imperative `loss.backward()` + `optimizer.step()` loops or FSDPv2. |

### Complete `torch.tpu.optimizers.SingleTraceTrainer` Recipe (`docs/public_apis.md`)

```python
import torch
from torch.tpu import optimizers

with torch.device("tpu"):
    model = MyTransformerModel().to(torch.bfloat16)

# Initialize SingleTraceTrainer with public functional AdamW (or optimizers.Muon / optimizers.Sgd)
# Note: optimizers.Muon(lr=1e-3, ns_steps=5, adjust_lr_fn="match_rms_adamw") runs 5-step Newton-Schulz
# orthogonalization on 2D hidden weights (requires unsharded 2D params; _SUPPORTS_SHARDED_PARAMS=False).
trainer = optimizers.SingleTraceTrainer(
    model=model,
    optimizer=optimizers.AdamW(lr=1e-4, weight_decay=0.01, use_bfloat16_moments=True),
    compile_fn=lambda gm, example_inputs: torch.compile(gm, backend="tpu", fullgraph=True),
)

# Returns a lazy single-trace step function that compiles FWD + Loss + BWD + OPT on its first call (0 graph breaks)
compiled_train_step = trainer.make_compiled_train_step()

for step, (inputs, targets) in enumerate(dataloader):
    loss = compiled_train_step(inputs, targets)
```

## 3. Production PEFT LoRA Fine-Tuning Checklist on TorchTPU

When fine-tuning `Llama-3.2-1B`, `Qwen3-1.7B`, or `Gemma-2-2B` with Hugging Face `peft` on TorchTPU (`google/prototypes/llama3/llama3_2_1b_training_peft_test.py`), enforce these four TPU-specific rules:

1. **Static Sequence Length Padding**: Always tokenize with `padding="max_length"`, `truncation=True`, and `DataLoader(..., drop_last=True)`. Variable batch or sequence dimensions trigger XLA recompilations.
2. **`num_workers=0` on `DataLoader`**: Never use `num_workers > 0` with the default `fork` start method after initializing `torch_tpu`; calling `os.fork()` after PJRT locks the TPU device causes child dataloader processes to hang (`docs/markdown/compilation_cache.md`).
3. **Set `lora_dropout=0.0`**: TorchTPU's `SDPBackend.OVERRIDEABLE` and fused linear paths require `dropout_p == 0.0` for maximum fusion.
4. **Use `capturable=True, fused=True` on `torch.optim.AdamW`**: Allows XLA to fuse the LoRA adapter parameter updates on device.

```python
import peft
import torch
import transformers

peft_config = peft.LoraConfig(
    task_type=peft.TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,  # Keep 0.0 for full XLA operator fusion on TPU
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
)
model = peft.get_peft_model(model, peft_config).to("tpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, capturable=True, fused=True)
```

## 4. Eliminating Logits HBM OOMs: Chunkwise Fused Unembedding Cross-Entropy

Modern LLMs have massive token vocabularies:
* **Llama-3**: `vocab_size = 128,256`
* **Qwen3**: `vocab_size = 151,936`
* **Gemma-2 / Gemma-3 / Gemma-4**: `vocab_size = 256,000`

At `batch_size = 4` and `seq_len = 8,192` ($32,768$ tokens per micro-batch), materializing the full `[4, 8192, 256000]` logits tensor in `float32` for Cross-Entropy loss requires:

$$\text{Logits Memory} = 4 \times 8,192 \times 256,000 \times 4\text{ bytes} = \mathbf{33.55\text{ GB of HBM!}}$$

On a `v6e-1` chip (`32 GB` total HBM), the logits tensor alone causes an immediate Out-Of-Memory crash!

### The TorchTPU Chunked Unembedding Pattern (`model_afmv7.py`)
Instead of projecting all $B \times S$ tokens to `[B*S, V]` at once, flatten the hidden state $H$ to `[N, D]` (where $N = B \times S$), reshape into chunks of `CHUNK_SIZE = 512` tokens, and compute `h_chunk @ lm_head_weight.T` + `F.cross_entropy` inside a loop captured by `@torch.compile(backend="tpu", dynamic=False)`:

```python
def chunked_unembed_cross_entropy(hidden_states, lm_head_weight, labels, chunk_size: int = 512):
    # Shift for causal next-token prediction: predict token t+1 from position t
    shift_h = hidden_states[:, :-1, :].contiguous().view(-1, hidden_states.size(-1))
    shift_labels = labels[:, 1:].contiguous().view(-1)

    # Pad token count to an exact multiple of chunk_size if needed
    num_tokens = shift_h.size(0)
    num_chunks = num_tokens // chunk_size
    h_chunks = shift_h.view(num_chunks, chunk_size, -1)
    l_chunks = shift_labels.view(num_chunks, chunk_size)

    total_loss = torch.zeros((), dtype=torch.float32, device=hidden_states.device)
    for i in range(num_chunks):
        # Materializes only [512, vocab_size] (~512 MB instead of 33.5 GB!)
        chunk_logits = (h_chunks[i] @ lm_head_weight.t()).float()
        total_loss = total_loss + torch.nn.functional.cross_entropy(
            chunk_logits, l_chunks[i], reduction="mean"
        )
    return total_loss / num_chunks
```

## 5. Guided Hands-On Lab (`labs/lab_02_single_trace_lora_finetuning.py`)

Open `labs/lab_02_single_trace_lora_finetuning.py` in the right-hand lab pane:
1. **Part 1 (Full Parameter vs. LoRA Memory & Parameter Accounting)**: Injects low-rank adapters ($W + \frac{\alpha}{r} B A$ with $r=8$) into a Transformer block and compares trainable parameter counts, optimizer state bytes, and gradient memory against full fine-tuning.
2. **Part 2 (Functional Single-Trace Training Step)**: Implements a functional zero-graph-break training step (`grad_and_value` + functional AdamW update) in both PyTorch (`torch.func`) and JAX (`jax.value_and_grad`) and trains for 15 steps.
3. **Part 3 (Chunkwise Fused Unembedding Cross-Entropy Verification)**: Verifies that chunked vocabulary cross-entropy produces **bit-exact identical loss and gradients** (`< 1e-6` diff) to full-tensor cross-entropy while reducing peak logits memory by **16x**.
