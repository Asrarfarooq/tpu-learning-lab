# TorchTPU & JAX Interactive Learning Lab

Self-paced technical training platform and live Cloud TPU code runner for PyTorch (`torch_tpu`), JAX, XLA compilation, XProf hardware profiling, and Pallas custom kernels.

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
* `scripts/install_tpu_runtime.sh`: Downloads `torch_tpu` wheels locally on Cloudtop into `.wheelhouse/` using your corporate Artifact Registry access, copies `.wheelhouse/` to the Argolis TPU VM via `gcloud compute tpus tpu-vm scp`, and installs Python 3.12, `torch`, `torch_tpu`, and `jax[tpu]` offline on the TPU VM.
