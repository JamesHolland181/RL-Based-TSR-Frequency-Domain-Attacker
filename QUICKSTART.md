# Quick Reference Guide

A cheat sheet for common tasks and commands in this project.

## 📥 Setup Commands

```bash
# Clone repository
git clone https://github.com/yourusername/RL-Based-TSR-Frequency-Domain-Attacker.git
cd RL-Based-TSR-Frequency-Domain-Attacker

# Setup environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_workflow.py
```

## 🚀 Common Tasks

### Generate Adversarial Dataset

```bash
cd TSR_Models
python generate_adv_dataset.py
```

**Customize generation:**
```python
# In generate_adv_dataset.py, modify:
generate_adversarial_dataset(
    dataset,
    num_samples=100,  # Number of samples
    patch_types=['occlusion', 'light'],  # Attack types
    size=20  # Patch size
)
```

### Test Model Loading

```bash
python test_models.py
```

### Train Model

```bash
cd TSR_Models
python train_and_eval.py
```

### Benchmark Models

```bash
cd TSR_Models
python benchmark_and_eval.py
```

## 🔧 Code Snippets

### Load Dataset

```python
from lisa import LISA

dataset = LISA(root='./data', train=True, download=True)
image, label = dataset[0]
print(f"Image shape: {image.shape}, Label: {label}")
```

### Load Pre-trained Model

```python
from models_factory import load_cnn_3spt, load_sag_vit
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_cnn_3spt(num_classes=47, device=device)
```

### Generate Single Adversarial Example

```python
from generate_adv_dataset import apply_adversarial_patch

# Apply attack
adv_image = apply_adversarial_patch(
    image,
    patch_type='occlusion',  # or 'light', 'perturbation', 'localized_perturbation'
    size=20
)
```

### Run Inference

```python
import torch

model.eval()
with torch.no_grad():
    image_batch = image.unsqueeze(0).to(device)  # Add batch dimension
    output = model(image_batch)
    prediction = torch.argmax(output, dim=1)
    print(f"Predicted class: {prediction.item()}")
```

### Visualize Results

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image.permute(1, 2, 0))
axes[0].set_title('Original')
axes[1].imshow(adv_image.permute(1, 2, 0))
axes[1].set_title('Adversarial')
plt.savefig('comparison.png')
```

## 🐛 Debugging Commands

### Check CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

### Check Model Parameters

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### Monitor GPU Memory

```python
import torch

if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
```

### Clear GPU Cache

```python
import torch
import gc

torch.cuda.empty_cache()
gc.collect()
```

## 📊 Model Information

| Model | Input Size | Parameters | Weights File |
|-------|-----------|------------|--------------|
| CNN-3SPT | 48×48 | 5.9M | `cnn_3spt.pth` |
| SAG-ViT | 224×224 | 6.7M | `sag_vit.pth` |
| ResNet50 | 224×224 | ~23M | `resnet_cnn_model.pth` |
| ViT-B/16 | 224×224 | ~86M | `vit_pretrained_best.pth` |

## 🎨 Attack Types

| Attack Type | Description | Use Case |
|------------|-------------|----------|
| `occlusion` | Random colored patch | Simulate physical obstruction |
| `light` | Bright reflection | Simulate glare/lighting issues |
| `perturbation` | Gaussian noise | Simulate sensor noise |
| `localized_perturbation` | Combined light + noise | Complex real-world scenarios |

## 📁 Important Directories

```
TSR_Models/          # Model implementations and training scripts
Model_Weights/       # Pre-trained model weights (Git LFS)
adversarial_dataset/ # Generated adversarial examples
data/               # LISA dataset (auto-downloaded)
```

## 🔑 Key Files

- `generate_adv_dataset.py` - Generate adversarial datasets
- `train_and_eval.py` - Train models
- `benchmark_and_eval.py` - Evaluate models
- `models_factory.py` - Model loading utilities
- `lisa.py` - LISA dataset loader
- `test_workflow.py` - Verify installation
- `test_models.py` - Test model loading

## 🌐 Git Commands

### Update from Upstream

```bash
git fetch upstream
git merge upstream/main
```

### Create Feature Branch

```bash
git checkout -b feature/my-feature
```

### Commit Changes

```bash
git add .
git commit -m "type(scope): description"
# Types: feat, fix, docs, style, refactor, test, chore
```

### Push to GitHub

```bash
git push origin feature/my-feature
```

## 🔍 Testing

```bash
# Run all tests
python test_workflow.py
python test_models.py

# Test specific module
python -m pytest tests/test_attacks.py

# Test with coverage
python -m pytest --cov=TSR_Models
```

## 📦 Export Models

### ONNX Export

```python
import torch.onnx

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

### TorchScript Export

```python
traced = torch.jit.trace(model, dummy_input)
traced.save("model_traced.pt")
```

## ⚡ Performance Tips

```python
# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    output = model(input)

# Increase batch size (if memory allows)
BATCH_SIZE = 64

# Use multiple workers for data loading
num_workers = 4

# Pin memory for faster GPU transfer
pin_memory = True
```

## 🛠️ Environment Variables

```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Set custom dataset path
export TSR_DATASET_PATH="./custom_data"

# Set model weights directory
export TSR_MODEL_WEIGHTS="./models"
```

## 📧 Quick Links

- [Full Documentation](README.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Development Guide](DEVELOPMENT.md)
- [Workflow Details](README_WORKFLOW.md)
- [Issues](https://github.com/yourusername/RL-Based-TSR-Frequency-Domain-Attacker/issues)
- [Pull Requests](https://github.com/yourusername/RL-Based-TSR-Frequency-Domain-Attacker/pulls)

---

**💡 Tip:** Bookmark this page for quick access to common commands!
