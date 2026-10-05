# Unit 2.2: Asynchronous Dispatch, Eliminating Host Stalls & TorchTPU Error Glossary

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Explain how **Asynchronous Host-Device Dispatch** overlaps Python host execution with TPU TensorCore execution.
2. Identify every Python operation that triggers a **Host-Device Synchronization Stall** (`loss.item()`, `print(tensor)`, `tensor.cpu()`, Python `if tensor:` branching).
3. Implement **Detached Device-Side Loss Accumulation** in TorchTPU and **`jax.lax.scan` Multi-Step Fusion** in JAX to eliminate host stalls.
4. Diagnose and fix the top production TorchTPU errors (`@torch.inference_mode()` crash with `torch.compile`, `TORCH_SHOW_CPP_STACKTRACES=1`, `TORCH_TPU_LAUNCH_BLOCKING=1`, and `OpTracer`).

## 2. How Asynchronous Dispatch Works on TPUs

Compiled TPU programs (`torch.compile` and `@jax.jit`) execute **asynchronously** with respect to the Python host thread:
* When Python calls `loss = compiled_step(batch)`, the call returns control to Python in microseconds **before the TPU has even started computing the step**.
* While the TPU is executing Step 1, Python runs ahead to prepare and enqueue Step 2, Step 3, and Step 4 into the `libtpu` device queue.
* As long as the device queue never empties, the TPU executes back-to-back steps with **zero idle gaps**.

```
BAD: Synchronous Per-Step Logging (loss.item() on every step)
Host CPU : [Enq 1] ----waits for TPU----> [Print] [Enq 2]
TPU Chip :        [=== Step 1 ===] (IDLE!)        [=== Step 2 ===]
                                   ^^^^^^^
                            Host-Sync Bubble!

GOOD: Asynchronous Dispatch (Log every K steps via device tensor)
Host CPU : [Enq 1][Enq 2][Enq 3][Enq 4] (Host runs ahead of TPU)
TPU Chip :        [== Step 1 ==][== Step 2 ==][== Step 3 ==]
                                 (0% Idle Gap!)
```

## 3. The Two Production Patterns for Stall-Free Training

### Pattern A: TorchTPU Detached On-Device Loss Accumulation
Never call `loss.item()` on every step. Instead, maintain a scalar tensor `running_loss` **on the TPU** and accumulate `loss.detach()` (so the autograd graph is not kept alive across steps). Only copy `running_loss.item()` to the CPU once every `LOG_INTERVAL` steps:

```python
running_loss = torch.zeros((), dtype=torch.float32, device=device)

for step, (x, y) in enumerate(dataloader, start=1):
    loss = compiled_train_step(model, optimizer, x, y)
    # Stays 100% on TPU HBM! Zero host-device synchronization!
    running_loss += loss.detach()

    if step % LOG_INTERVAL == 0:
        # Only 1 host sync every LOG_INTERVAL steps
        avg_loss = (running_loss / LOG_INTERVAL).item()
        running_loss.zero_()
        print(f"Step {step} | Avg Loss: {avg_loss:.5f}")
```

### Pattern B: JAX `jax.lax.scan` Multi-Step Loop Fusion
In JAX, you can go one step further: instead of launching $K$ separate steps from a Python `for` loop, `jax.lax.scan` compiles **$K$ training steps into a single XLA `while` loop** that runs entirely on the TPU with zero Python interaction for all $K$ steps:

```python
@jax.jit
def train_k_steps(params, opt_state, batch_sequence):
    def body_fn(carry, batch):
        p, o_state = carry
        x_b, y_b = batch
        loss_val, grads = jax.value_and_grad(loss_fn)(p, x_b, y_b)
        p, o_state = update_params(p, o_state, grads)
        return (p, o_state), loss_val

    # Executes all K steps inside one XLA call on the TPU!
    (final_params, final_opt_state), per_step_losses = jax.lax.scan(
        body_fn, (params, opt_state), batch_sequence
    )
    return final_params, final_opt_state, jnp.mean(per_step_losses)
```

## 4. Production Debugging Playbook: The TorchTPU Error Glossary (`go/tt-devguide`)

When debugging errors in TorchTPU, **asynchronous deferred execution means an exception may be raised many lines after the offending operation** (specifically, at the next `print()`, `.cpu()`, or `.item()` sync barrier). Use these verified techniques from the internal TorchTPU Error Glossary:

| Symptom / Error Message | Root Cause | Verified Fix |
| :--- | :--- | :--- |
| **`RuntimeError: Inference tensors cannot be saved for backward` or AOTAutograd crash under `torch.compile`** | Decorating an evaluation function with **`@torch.inference_mode()`** while using `torch.compile(backend="tpu")`. Inference mode disables version counters required by TorchDynamo/AOTAutograd graph tracing. | Replace `@torch.inference_mode()` with **`@torch.no_grad()`** whenever using `torch.compile(backend="tpu")`. |
| **`NotImplementedError: Could not run 'aten::...' with arguments from the 'TPU' backend`** | An operator has no lowered `TT_KERNEL` in C++ yet and was called on a TPU tensor. | Wrap the specific unsupported pre/post-processing op in `.cpu()` -> `.to("tpu")`, or implement it via a JAX/Pallas custom op (`torch.tpu.pallas.jax_op`). |
| **Stack trace points to `print(tensor)` instead of the broken op** | In Defer-and-Fuse mode (`TORCH_TPU_DEFER_AND_FUSE=1`), compilation waits until `print(tensor)` flushes the queue. | Set **`TORCH_TPU_LAUNCH_BLOCKING=1`** (or `TORCH_TPU_DEFER_AND_FUSE=0`) and **`TORCH_SHOW_CPP_STACKTRACES=1`** before `import torch` to pinpoint the exact failing Python line and C++ frame. |
| **Inspecting which ATen ops were dispatched** | Need to see exact input/output shapes and dtypes of every ATen op inside a block. | Wrap the block in `from torch_tpu._internal.utils.utils import OpTracer; with OpTracer(): ...` |

## 5. Guided Hands-On Lab (`labs/lab_02_stall_free_training_loop.py`)

Inspect `labs/lab_02_stall_free_training_loop.py` in the right-hand code editor:
1. **Part 1 (TorchTPU)**: Benchmarks 40 training steps comparing **Anti-Pattern (per-step `loss.item()` sync)** against **Best Practice (on-device `running_loss += loss.detach()` synced once every 40 steps)**.
2. **Part 2 (JAX)**: Benchmarks 40 training steps comparing a **Python `for` loop with per-step `float(loss)`** against **`jax.lax.scan` fusing 40 steps per XLA call**.

Click **Run Code** to measure the throughput speedup from eliminating host stalls.
