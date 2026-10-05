# Unit 0.3: Cloud TPU VM Architecture, Topologies & Spot Provisioning

## 1. Learning Objectives & Production Context

By the end of this unit, you will be able to:
1. Explain the **Cloud TPU VM (1VM)** direct-attach architecture vs. legacy 2VM gRPC proxy nodes.
2. Decode GCP TPU accelerator topology names (`v6e-1`, `v6e-4`, `v6e-8`, `v5p-8`, `v5litepod-4`) in terms of **physical chips**, **TensorCores per chip**, **HBM capacity**, **Inter-Chip Interconnect (ICI)** topology, and **VM host count**.
3. Provision a **Spot Cloud TPU VM** in `torchtpu-learning-lab` using Cloud TPU Queued Resources and stage `torch_tpu` + `jax[tpu]` without exposing corporate `@google.com` tokens on the external Argolis VM.

## 2. Cloud TPU VM (1VM) Architecture

Historically (TPU v2/v3 prior to 2021), Cloud TPUs used a "2VM" split architecture where users logged into a standard GCE VM and sent XLA programs over gRPC to a separate, inaccessible TPU host.

Modern Cloud TPUs (**TPU v4, v5e, v5p, and v6e Trillium**) exclusively use the **Cloud TPU VM (1VM)** architecture:
* When you provision a TPU VM, you receive **direct SSH access to the physical Linux host machine** whose PCIe root complex is wired directly to the TPU boards.
* Your Python process (`python3`) loads the shared library **`libtpu.so`** into its address space.
* Both **JAX** (`jaxlib` PJRT client) and **TorchTPU** (`torch_tpu` PJRT client) talk to the exact same `libtpu.so` C API (`PJRT` — Plug-in PJRT Runtime) to compile StableHLO graphs and drive the TPU hardware via kernel driver `/dev/accel*` or `/dev/vfio/*`.

```
+-------------------------------------------------------------------------+
|                    CLOUD TPU VM HOST (Ubuntu 22.04)                     |
|                                                                         |
|   +-------------------------------+   +-----------------------------+   |
|   | PyTorch Process               |   | JAX Process                 |   |
|   | (torch + torch_tpu)           |   | (jax + jaxlib)              |   |
|   +---------------+---------------+   +--------------+--------------+   |
|                   |                                  |                  |
|                   +----------------+-----------------+                  |
|                                    | PJRT C API (StableHLO -> HLO/LLO)  |
|                                    v                                    |
|                         +---------------------+                         |
|                         |      libtpu.so      |                         |
|                         | (XLA TPU Compiler & |                         |
|                         |  Runtime Driver)    |                         |
|                         +----------+----------+                         |
|                                    | Direct PCIe / VFIO (/dev/vfio/*)   |
+------------------------------------+------------------------------------+
                                     |
        +----------------------------+----------------------------+
        |                            |                            |
        v                            v                            v
+---------------+            +---------------+            +---------------+
|  TPU Chip 0   | <========> |  TPU Chip 1   | <========> |  TPU Chip 2   | ...
| (32-95GB HBM) |  ICI Link  | (32-95GB HBM) |  ICI Link  | (32-95GB HBM) |
+---------------+            +---------------+            +---------------+
```

### Important Operational Consequence: Single-Process Hardware Lock
Because `libtpu.so` maps the TPU PCIe VFIO device files directly into the process memory space for zero-copy DMA, **only one OS process can hold the TPU device lock on a given chip at a time**.
* If Process A is holding `/dev/vfio/0` (for example, an idle Python REPL or background zombie training script), Process B will fail on startup with `RESOURCE_EXHAUSTED: Accelerator device is busy`.
* Our lab runner executes each script in a clean, short-lived process that releases the TPU lock immediately upon completion.

## 3. Decoding TPU Generations & Topologies (`v6e` Trillium vs. `v5p` vs. `v5e`)

| Spec / Metric | TPU v6e (Trillium) | TPU v5p | TPU v5e (`v5litepod`) |
| :--- | :--- | :--- | :--- |
| **Target Workload** | High-efficiency training & low-latency LLM serving | Massive-scale frontier LLM training | Cost-optimized medium training & inference |
| **TensorCores per Chip** | **1 TensorCore** per chip | **2 TensorCores** per chip (MegaCore fused) | **1 TensorCore** per chip |
| **MXU Systolic Array Size** | **$256 \times 256$** (2 MXUs / chip) | **$128 \times 128$** (4 MXUs / core, 8 / chip) | **$128 \times 128$** (4 MXUs / chip) |
| **Peak `bf16` TFLOPs / Chip** | **918 TFLOPs** | **459 TFLOPs** | **197 TFLOPs** |
| **HBM Capacity / Chip** | **32 GB** | **95 GB** | **16 GB** |
| **HBM Bandwidth / Chip** | **1,640 GB/s** | **2,765 GB/s** | **820 GB/s** |
| **Single-Host Topologies** | `v6e-1` (1 chip), `v6e-4` (4 chips, $2\times 2$), `v6e-8` (8 chips, $2\times 4$) | `v5p-8` (4 chips / 8 cores, $2\times 2\times 1$) | `v5litepod-1` (1 chip), `v5litepod-4` (4 chips, $2\times 2$), `v5litepod-8` (8 chips, $2\times 4$) |
| **GCP Runtime Version** | `v2-alpha-tpuv6e` | `v2-alpha-tpuv5` | `v2-alpha-tpuv5-lite` |

### Why `v5p-8` Has 4 Logical Devices (`device_count() == 4`)
Notice the naming difference between `v5p` and `v6e`:
* On **TPU v5p**, each physical silicon die contains **2 TensorCores**. GCP names `v5p` slices by **TensorCore count** (`v5p-8` = 8 TensorCores = 4 physical chips on 1 VM). By default, `libtpu` runs in **MegaCore** mode, which binds the 2 TensorCores on each chip together so they share the chip's 95 GB HBM as a single logical device (380 GB total HBM on `v5p-8`). Therefore, both `torch.accelerator.device_count()` and `jax.device_count()` return **`4`** on a `v5p-8` VM.
* On **TPU v6e (Trillium)**, Google redesigned the chip around **1 giant TensorCore** with two $256 \times 256$ MXUs and 32 GB HBM. Therefore, `v6e-<N>` names the **exact number of physical chips** (`v6e-1` = 1 chip, `v6e-4` = 4 chips, `v6e-8` = 8 chips).

## 4. Provisioning & Wheelhouse Staging in `torchtpu-learning-lab`

You can manage your Spot TPU VM either from the **TPU Infrastructure** drawer at the top of this web app or directly from your terminal using the three scripts in `/usr/local/google/home/asrarfarooq/tpu-learning-lab/scripts/`:

### 1. Bootstrap VPC & Check Quota (`scripts/bootstrap_project.sh`)
```bash
./scripts/bootstrap_project.sh us-central1
```
Enables `compute.googleapis.com` and `tpu.googleapis.com` on `torchtpu-learning-lab` and creates `tpu-lab-vpc` (`MTU 8896`) with SSH/ICI firewall rules.

### 2. Provision Spot TPU VM via Queued Resources (`scripts/provision_spot_tpu.sh`)
```bash
# Request a Spot v6e-1 in us-central1-b (or v5p-8 in us-central1-b / us-east5-a)
./scripts/provision_spot_tpu.sh create us-central1-b v6e-1 tpu-lab-vm

# Check provisioning status
./scripts/provision_spot_tpu.sh status us-central1-b
```

### 3. Stage Local `.wheelhouse/` and Install Runtime (`scripts/install_tpu_runtime.sh`)
```bash
./scripts/install_tpu_runtime.sh us-central1-b tpu-lab-vm
```
This script:
1. Uses your local Cloudtop corporate credentials (`asrarfarooq@google.com`) to run `pip download` into `/usr/local/google/home/asrarfarooq/tpu-learning-lab/.wheelhouse/` from `us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/`.
2. Copies `.wheelhouse/` to `tpu-lab-vm:~/wheelhouse/` using `gcloud compute tpus tpu-vm scp --project=torchtpu-learning-lab`.
3. Installs Python 3.12, `torch_tpu` (from `~/wheelhouse`), and `jax[tpu]` inside `~/tpu-env` on the TPU VM.
