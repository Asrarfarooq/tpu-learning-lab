# Unit 2.1: Neural Modules, RMSNorm, SwiGLU, ViT Porting & Optimizer State

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Build modern Transformer building blocks (**RMSNorm**, **SwiGLU MLP**, and **Vision Transformer `TpuViT`**) side-by-side in **PyTorch (`torch.nn.Module`)** and **JAX (PyTrees / Flax)**.
2. Calculate the exact **HBM Memory Budget** of training: Model Weights + Gradients + **AdamW Optimizer State ($m_t, v_t$)** + Forward Activations saved for Backward.
3. Configure `torch.utils.data.DataLoader` safely for Cloud TPU VMs (`drop_last=True`, `num_workers=0`) and avoid `F.scaled_dot_product_attention` math-fallback pitfalls.

## 2. Modern LLM Building Blocks: RMSNorm & SwiGLU

Modern frontier models (Llama-3, Gemma, PaLM/Gemini) replace classic `LayerNorm` and `ReLU` with **RMSNorm** and **SwiGLU**:

### 1. Root Mean Square Layer Normalization (RMSNorm)
Standard `LayerNorm` computes both the mean $\mu$ and variance $\sigma^2$ across the hidden dimension $D$. **RMSNorm** omits mean-centering and normalizes solely by the Root Mean Square, reducing vector reduction overhead on the TPU VPU:

$$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2 + \epsilon}, \qquad \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma$$

* **Why Upcast to `float32` Inside `RMSNorm`?**
  When activations $x$ are stored in `bfloat16`, squaring and summing $D=4096$ elements can lose precision or overflow in `bfloat16`. Always compute the mean-square reduction in **`float32`** (`x.float().pow(2).mean(-1, keepdim=True)`), then cast back to `x.dtype`.

### 2. SwiGLU Feed-Forward Network
Instead of a 2-matrix MLP $\text{GELU}(x W_1) W_2$, **SwiGLU** uses a gated projection with three weight matrices ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$):

$$\text{SwiGLU}(x) = \Big(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\Big) W_{\text{down}}, \qquad \text{where } \text{SiLU}(z) = z \cdot \sigma(z)$$

## 3. Training Memory Budget: Where Does TPU HBM Actually Go?

A common surprise for engineers new to model training is that **a 7-billion-parameter model does not take 14 GB of HBM to train—it takes ~84 to 112 GB of HBM before activations!** Here is the exact byte accounting per parameter $\theta$:

| Tensor Buffer in HBM | Precision in Mixed-Precision Training | Bytes per Parameter | Size for a 7B Parameter Model |
| :--- | :--- | :--- | :--- |
| **1. Master Weights ($\theta$)** | `float32` (or `bfloat16`) | `4 bytes` (or `2 bytes`) | `28 GB` (or `14 GB`) |
| **2. Gradients ($\nabla_\theta L$)** | `bfloat16` or `float32` | `2 bytes` or `4 bytes` | `14 GB` – `28 GB` |
| **3. AdamW 1st Moment ($m_t$)** | `float32` | `4 bytes` | `28 GB` |
| **4. AdamW 2nd Moment ($v_t$)** | `float32` | `4 bytes` | `28 GB` |
| **Total Static State** | **Weights + Grads + AdamW** | **12 to 16 bytes / param** | **84 GB to 112 GB** |

### Why AdamW Needs $m_t$ and $v_t$
Unlike plain Stochastic Gradient Descent ($W \leftarrow W - \eta g_t$, which requires **0 bytes** of optimizer state), **AdamW** adapts the learning rate for every individual weight using two running exponential moving averages:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \qquad (\text{1st moment: momentum})$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \qquad (\text{2nd moment: per-weight variance})$$
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

## 4. Porting Models to TorchTPU: Key Lessons from `TpuViT` (`go/tt-devguide`)

When porting PyTorch architectures (such as a Vision Transformer `ViT` or custom Encoder/Decoder) to `torch_tpu`:

1. **Patchify via `nn.Conv2d(..., kernel_size=P, stride=P)` + `.flatten(2).transpose(1, 2)`**:
   Maps cleanly to TPU's native convolution and transpose engines (`XLU`) without Python loops.
2. **Explicit Multi-Head Attention (`(Q @ K.transpose(-2, -1)) * scale`)**:
   Stock `nn.TransformerEncoderLayer` can route through CUDA-specific fastpaths or math fallbacks depending on mask types. Writing explicit batched matrix multiplication (`q @ k.transpose(-2, -1)` followed by `F.softmax(..., dim=-1) @ v`) guarantees 100% XLA `DotGeneral` lowering onto the TPU MXU without graph breaks.
3. **`DataLoader(..., drop_last=True, num_workers=0)`**:
   * **`drop_last=True`**: Ensures the final batch of an epoch never has a smaller remainder batch size (e.g., `16` instead of `64`), which would trigger a multi-second XLA recompilation right at the end of the epoch.
   * **`num_workers=0`**: Avoids POSIX `fork()` multiprocessing deadlocks with `libtpu.so` background threads.

## 5. Guided Hands-On Lab (`labs/lab_01_mlp_and_vit_block.py`)

Inspect `labs/lab_01_mlp_and_vit_block.py` in the right-hand code editor:
1. It implements a complete **Pre-RMSNorm + SwiGLU Transformer MLP Block** in both **PyTorch (`nn.Module` + `torch.optim.AdamW`)** and **JAX (Pure PyTree + explicit AdamW state)**.
2. It initializes both models with identical deterministic weights and verifies forward-pass numerical parity (`< 1e-5`).
3. It audits the exact byte footprint of parameters, gradients, and AdamW state buffers ($m_t, v_t$), and trains both blocks for 15 steps.

Click **Run Code** to execute the lab.
