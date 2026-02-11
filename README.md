# RL-Based Traffic Sign Recognition with Frequency Domain Adversarial Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A comprehensive framework for developing and evaluating robust Traffic Sign Recognition (TSR) systems using reinforcement learning-based adversarial training in the frequency domain. This repository includes multiple state-of-the-art model architectures, adversarial attack generation, and transfer learning capabilities.

## 🌟 Features

- **Multiple TSR Model Architectures**:
  - CNN-3SPT (Spatial Transformer Network)
  - SAG-ViT (Spatial-Aware Graph Vision Transformer)
  - ResNet50 (Transfer Learning)
  - ViT-B/16 (Vision Transformer)

- **Adversarial Attack Generation**:
  - Occlusion attacks
  - Light reflection simulation
  - Gaussian noise perturbations
  - Localized perturbations (combined attacks)

- **Automated Workflows**:
  - Dataset generation and management
  - Model benchmarking and evaluation
  - Transfer learning and fine-tuning
  - Performance visualization

- **Pre-trained Model Weights**: Production-ready models trained on LISA dataset

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🔧 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 2GB+ available disk space
- Internet connection (for dataset download)

### Required Python Packages

- PyTorch >= 2.0
- torchvision
- torch-geometric
- timm
- scikit-learn
- pandas
- matplotlib
- opencv-python
- numpy

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/RL-Based-TSR-Frequency-Domain-Attacker.git
cd RL-Based-TSR-Frequency-Domain-Attacker
```

### 2. Set Up Python Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If `requirements.txt` is not available, install manually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric scikit-learn pandas matplotlib opencv-python numpy timm
```

### 4. Verify Installation

```bash
python test_workflow.py
```

Expected output should show:
- ✓ All dependencies installed
- ✓ LISA dataset loaded
- ✓ Adversarial patch generation works
- ✓ Model weights verified
- ✓ Model architectures loadable

## ⚡ Quick Start

### Generate Adversarial Dataset

```bash
cd TSR_Models
python generate_adv_dataset.py
```

This will:
- Download LISA dataset (if not present)
- Generate 50 original images
- Create 200 adversarial variants (4 attack types × 50 images)
- Save to `adversarial_dataset/`

### Benchmark Pre-trained Models

```bash
python test_models.py
```

This tests model loading and provides performance metrics.

### Train with Transfer Learning

```bash
cd TSR_Models
python train_and_eval.py
```

This fine-tunes pre-trained models on your dataset.

## 📁 Project Structure

```
RL-Based-TSR-Frequency-Domain-Attacker/
├── TSR_Models/                      # Main model implementations
│   ├── generate_adv_dataset.py     # Generate adversarial datasets
│   ├── benchmark_and_eval.py       # Model benchmarking
│   ├── train_and_eval.py           # Transfer learning pipeline
│   ├── cnn_3spt.py                 # CNN-3SPT architecture
│   ├── SAG_VIT.py                  # SAG-ViT architecture
│   ├── models_factory.py           # Model loader utilities
│   ├── lisa.py                     # LISA dataset loader
│   ├── custom_data.py              # Custom dataset utilities
│   ├── model_components.py         # Shared model components
│   └── graph_construction.py       # Graph construction for SAG-ViT
│
├── Adversarial_Training/            # Alternative training scripts
│   ├── AdversarialPatches4.py      # Adversarial patch generation
│   ├── lisa.py                     # LISA dataset loader
│   ├── model_components.py         # Model building blocks
│   ├── graph_construction.py       # Graph utilities
│   └── PNGDataset.py               # PNG dataset loader
│
├── Model_Weights/                   # Pre-trained model weights (Git LFS)
│   ├── cnn_3spt.pth               # CNN-3SPT (22.4 MB)
│   ├── sag_vit.pth                # SAG-ViT (25.9 MB)
│   ├── vit_pretrained_best.pth    # ViT (327.5 MB)
│   └── resnet_cnn_model.pth       # ResNet50 (42.8 MB)
│
├── adversarial_dataset/            # Generated adversarial datasets
│   ├── train/                     # Original training images
│   ├── test/                      # Adversarial test images
│   └── labels.json                # Label mappings
│
├── data/                           # LISA dataset (auto-downloaded)
│   └── lisa-batches/              # Processed LISA data
│
├── test_workflow.py                # Workflow verification script
├── test_models.py                  # Model testing script
│
├── README.md                       # This file
├── README_WORKFLOW.md              # Detailed workflow documentation
├── SETUP_INSTRUCTIONS.md           # Setup guide
├── CONTRIBUTING.md                 # Contribution guidelines
├── requirements.txt                # Python dependencies
└── LICENSE                         # MIT License
```

## 📖 Usage

### 1. Working with Custom Datasets

```python
from lisa import LISA

# Load LISA dataset
dataset = LISA(root='./data', train=True, download=True)

# Access samples
image, label = dataset[0]  # Returns (Tensor[3, 32, 32], int)
```

### 2. Generating Adversarial Examples

```python
from generate_adv_dataset import apply_adversarial_patch

# Apply occlusion attack
adversarial_image = apply_adversarial_patch(
    image, 
    patch_type='occlusion',  # or 'light', 'perturbation', 'localized_perturbation'
    size=20
)
```

### 3. Loading Pre-trained Models

```python
from models_factory import load_cnn_3spt, load_sag_vit
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CNN-3SPT
model = load_cnn_3spt(num_classes=47, device=device)

# Load SAG-ViT
model = load_sag_vit(num_classes=47, device=device)
```

### 4. Model Inference

```python
import torch
from torchvision import transforms

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Run inference
model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0))
    prediction = torch.argmax(output, dim=1)
```

## 🏗️ Model Architectures

### CNN-3SPT
- **Input Size:** 48×48 RGB
- **Parameters:** 5.9M
- **Throughput:** ~55 images/sec (CPU)
- **Features:** Spatial Transformer Network, 3-stage processing

### SAG-ViT (Spatial-Aware Graph Vision Transformer)
- **Input Size:** 224×224 RGB
- **Parameters:** 6.7M
- **Throughput:** ~27 images/sec (CPU)
- **Features:** Graph attention, transformer encoder, patch-based processing

### ResNet50
- **Input Size:** 224×224 RGB
- **Parameters:** ~23M
- **Throughput:** ~40 images/sec (CPU)
- **Backbone:** ImageNet pre-trained

### ViT-B/16
- **Input Size:** 224×224 RGB
- **Parameters:** ~86M
- **Throughput:** ~20 images/sec (CPU)
- **Backbone:** ImageNet pre-trained

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test complete workflow
python test_workflow.py

# Test model loading and inference
python test_models.py

# Test adversarial generation
cd TSR_Models
python generate_adv_dataset.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Setting up your development environment
- Code style and standards
- Submitting pull requests
- Reporting issues
- Adding new features

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📊 Dataset Information

**LISA Traffic Sign Dataset:**
- 47 traffic sign classes (US road signs)
- 6,621 training samples
- ~1,300 test samples
- 32×32 RGB images
- Automatically downloaded on first use

**Adversarial Dataset:**
- Customizable attack types
- Configurable patch sizes
- Train/test split organization
- JSON label mappings

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size or switch to CPU
DEVICE = torch.device("cpu")
```

**2. Module Import Errors**
```bash
# Ensure virtual environment is activated
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac
```

**3. Dataset Download Fails**
```bash
# Check internet connection and retry
# Dataset will auto-download (26.4 MB)
```

**4. Model Weights Not Found**
- Ensure Git LFS is installed: `git lfs install`
- Pull LFS files: `git lfs pull`

For more issues, see [README_WORKFLOW.md](README_WORKFLOW.md#troubleshooting)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

James Holland - [GitHub Profile](https://github.com/yourusername)

Project Link: [https://github.com/yourusername/RL-Based-TSR-Frequency-Domain-Attacker](https://github.com/yourusername/RL-Based-TSR-Frequency-Domain-Attacker)

## 🙏 Acknowledgments

- LISA Traffic Sign Dataset from the Laboratory for Intelligent & Safe Automobiles
- PyTorch and timm library developers
- Open-source community contributors

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{tsr_adversarial_2026,
  author = {Holland, James},
  title = {RL-Based Traffic Sign Recognition with Frequency Domain Adversarial Attacks},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/RL-Based-TSR-Frequency-Domain-Attacker}
}
```

---

**Status:** ✅ Production Ready (February 2026)  
**Tested:** Python 3.8-3.13, PyTorch 2.0+  
**Platform:** Windows, Linux, macOS
