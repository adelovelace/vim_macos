<div align="center">
<h1>Vision Mamba (Vim) on Apple Silicon 🍎</h1>
<h3>Efficient Visual Representation Learning running on M1/M2/M3/M4 Chips via MPS</h3>

[Original Paper](https://arxiv.org/abs/2401.09417) | [Original Repository](https://github.com/hustvl/Vim)

</div>

#

This repository contains a ported version of **Vision Mamba (Vim)** adapted to run natively on macOS devices using Apple Silicon (M-series chips) via **Metal Performance Shaders (MPS)**.

The original codebase relies heavily on NVIDIA-specific CUDA kernels (Triton, fused operations). This adaptation implements PyTorch-native fallbacks for convolution and selective scan operations, allowing training and inference without NVIDIA GPUs.

## 🚀 Installation on macOS

Standard installation methods for Mamba fail on macOS due to `triton` and CUDA compiler requirements. Follow these steps to set up the environment on an M4 Max (or other Apple Silicon Macs).

### 1. Prerequisites
Ensure you have Python installed (Python 3.10+ recommended) and a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv vim_env
source vim_env/bin/activate

### 2. Install Dependencies

Install PyTorch (with MPS support) and other core libraries:

```bash
pip install torch torchvision torchaudio
pip install timm einops mlflow

```

### 3. Install Mamba and Causal-Conv1d (CPU/MPS Mode)

You must install these packages from the source code included in this repo, forcing them to skip the CUDA build process.

**A. Install mamba-ssm:**

```bash
cd mamba-1p1p1
# IMPORTANT: Use this flag to skip CUDA compilation
MAMBA_SKIP_CUDA_BUILD=TRUE pip install -e . --no-build-isolation
cd ..

```

**B. Install causal-conv1d:**

```bash
cd causal-conv1d
# IMPORTANT: Use this flag to skip CUDA compilation
CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE pip install -e . --no-build-isolation
cd ..

```

---

## 🏃 Usage

To run training or inference, you must specify the **mps** device and disable pinned memory (which can cause warnings on MPS).

### Quick Start (Imagenette)

To test the model on a smaller dataset like Imagenette:

```bash
python main.py \
  --model vim_tiny_patch16_224 \
  --batch-size 32 \
  --device mps \
  --data-path ./data/imagenette2 \
  --no-pin-mem

```

### Key Arguments for Mac

* `--device mps`: Forces the use of the Apple Silicon GPU.
* `--no-pin-mem`: Disables pinned memory (recommended for current MPS implementations).
* `--batch-size`: Adjust based on your Unified Memory (RAM). 32 or 64 is usually safe for M4 Max.

---

## 🛠️ Summary of Adaptations

To make this code compatible with Apple Silicon, the following changes were made to the original CUDA-based implementation:

### 1. Core Mamba Logic (`mamba_ssm`)

* **Manual Convolution:** Implemented a pure PyTorch fallback for `causal_conv1d_fn` using `F.conv1d`.
* **Dimension Fix:** Added logic to automatically reshape 2D weights to 3D `(D, 1, K)` to satisfy PyTorch CPU/MPS convolution requirements.
* **Reference Scan:** Switched to `selective_scan_ref` (Python implementation) instead of the unavailable `selective_scan_fn` CUDA kernel.
* **Safe Imports:** Wrapped CUDA imports in `try-except` blocks to prevent crashes on startup.

### 2. Model Architecture (`vim/models_mamba.py`)

* **RMSNorm Fallback:** Replaced Triton-based RMSNorm with a custom `nn.Module` implementation.
* **Fused Add Norm Safety:** Added a safety switch in `VisionMamba.__init__`. It automatically disables `fused_add_norm` if optimized kernels are missing, preventing runtime errors.
* **Timm Compatibility:** Updated deprecated imports (`timm.models.layers` -> `timm.layers`) for compatibility with modern PyTorch versions.

### 3. Engine & Utilities

* **MPS Synchronization:** Modified `engine.py` to check for the device type before calling synchronization. It now uses `torch.mps.synchronize()` instead of crashing on `torch.cuda.synchronize()`.
* **Memory Logging:** Updated `utils.py` to avoid querying CUDA memory stats when running on Mac.

---

## Citation

If you use this project, please cite the original paper:

```bibtex
@article{zhu2024vision,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2401.09417},
  year={2024}
}

```

```

```
