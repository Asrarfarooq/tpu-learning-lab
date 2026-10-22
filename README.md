# TorchTPU & JAX Interactive Learning Lab

Self-paced technical training platform and live Cloud TPU code runner for PyTorch (`torch_tpu` 0.2.0+ Open Source release), JAX, XLA compilation, XProf hardware profiling, and Pallas custom kernels.

## Quick Start

1. Start the local course platform (listens on `http://127.0.0.1:8765`, scoped strictly to GCP project `torchtpu-learning-lab`):
   ```bash
   ./run.sh
   ```
2. Open `http://127.0.0.1:8765` in your browser.

## Infrastructure Scripts (`scripts/`)

All scripts explicitly pass `--project=torchtpu-learning-lab` and never modify your global `gcloud` configuration:

* `scripts/bootstrap_project.sh`: Enables `compute.googleapis.com` and `tpu.googleapis.com`, configures the `tpu-lab-vpc` network and IAP/SSH firewall rule, and checks Spot TPU quota (`v6e` and `v5p`).
* `scripts/provision_spot_tpu.sh`: Requests or manages a Spot Cloud TPU VM (`v6e-1`, `v6e-4`, or `v5p-8`) via Cloud TPU Queued Resources (`--spot`).
* `scripts/install_tpu_runtime.sh`: Installs Python 3.12, public PyPI `torch_tpu==0.2.0` (`pip install torch_tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu`), and `jax[tpu]` directly on the TPU VM inside `~/tpu-env` (or pass `--from-wheelhouse` as the 3rd argument to install custom Bazel-built wheels from `.wheelhouse/`).
