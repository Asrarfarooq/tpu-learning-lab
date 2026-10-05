# Unit 4.2: Integrating Pallas Kernels into TorchTPU (`torch.tpu.pallas.jax_op`) & LLO Inspection

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Embed any JAX or Pallas function directly inside a **PyTorch `torch.compile(backend="tpu")`** model with **zero host-device copies** using `torch.tpu.pallas.jax_op("namespace::op_name", jax_fn)`.
2. Understand the production **`quant_and_pack`** case study from the TorchTPU Developer Guide (`go/tt-devguide`): why standard XLA `torch.compile` spills sub-32-bit bit-shifts to VMEM, and how Pallas (`jax.experimental.pallas.tpu` / `pltpu.bitcast`) compiles register-level packing into **3 TPU LLO hardware instructions** (`ftoi.rn.bf16`, `pack.b8`, `vst`).
3. Dump and inspect low-level TPU **LLO VLIW assembly** using `LIBTPU_INIT_ARGS="--xla_jf_dump_to=/tmp/llo_dump --xla_jf_dump_llo_text=true"`.

## 2. How `torch.tpu.pallas.jax_op` Bridges PyTorch and JAX at the StableHLO Level

Because both **TorchTPU** and **JAX** lower their operations to **StableHLO** MLIR before passing the graph to `libtpu.so`, TorchTPU provides a direct zero-copy bridge (`torch.tpu.pallas.jax_op`):

```python
import torch
import torch_tpu
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# 1. Define a JAX / Pallas function (using pltpu.bitcast / register-level bit packing):
def quant_and_pack_kernel(x_ref, out_ref):
    # Pack 32 binary signs into a 32-bit integer word entirely in VREGs:
    packed = jnp.zeros(out_ref.shape, dtype=jnp.int32)
    for k in range(32):
        bits = (x_ref[:, k::32] >= 0.0).astype(jnp.int32)
        packed = packed | jnp.left_shift(bits, k)
    out_ref[...] = packed

# 2. Register and wrap as a native PyTorch op in a single call!
torch_quant_and_pack = torch.tpu.pallas.jax_op("my_ops::quant_and_pack", quant_and_pack_jax)
```

### What `torch.tpu.pallas.jax_op("my_ops::quant_and_pack", jax_fn)` Does Automatically
1. Registers `"my_ops::quant_and_pack"` with `torch.library`.
2. Generates the `@register_fake` abstract shape/dtype inference rule by tracing `jax_fn` with `jax.eval_shape`.
3. Lowers the Pallas kernel to a Mosaic/StableHLO custom call inside `torch.compile(backend="tpu")` with **zero CPU round-trips**.

## 3. Verifying Kernel Quality via TPU LLO Assembly Dumps

To inspect the exact VLIW assembly instructions (**LLO — Low-Level Optimizer**) emitted by the TPU compiler, configure `LIBTPU_INIT_ARGS` **before** importing `torch`:

```python
import os, sys
os.environ["LIBTPU_INIT_ARGS"] = "--xla_jf_dump_to=/tmp/llo_dump --xla_jf_dump_llo_text=true"
sys.argv.extend(["--xla_jf_dump_to=/tmp/llo_dump", "--xla_jf_dump_llo_text=true"])
import torch
import torch_tpu
```

In the generated `*llo.txt` file, the Pallas quantized packing kernel compiles each tile iteration into just **3 hardware instructions**:
1. **`vs32 = ftoi.rn.bf16`** (Vector float-to-integer round-to-nearest conversion in registers)
2. **`pack.b8`** (Hardware byte-packing of four 8-bit registers into one 32-bit register)
3. **`vst`** (Single contiguous 32-bit vector store to VMEM)

## 4. Guided Hands-On Lab (`labs/lab_02_pallas_in_torch_compile.py`)

Click **Run Code** to execute `labs/lab_02_pallas_in_torch_compile.py`. It implements a **Fused Pallas RMSNorm Kernel**, bridges it into PyTorch via `torch.tpu.pallas.jax_op` / `torch.library.custom_op`, and verifies numerical parity between PyTorch and JAX Pallas.
