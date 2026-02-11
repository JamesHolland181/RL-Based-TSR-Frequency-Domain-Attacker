# TSR Workflow - Complete Traffic Sign Recognition System

## ✅ Verified Working (February 2026)

This repository contains a complete workflow for Traffic Sign Recognition using adversarial training techniques. All components have been tested and verified working.

## 📁 Directory Structure

```
TSR_Workflow_Complete_2026-02-08/
├── .venv/                          # Python virtual environment (pre-configured)
├── Model_Weights/                  # Pre-trained model weights
│   ├── cnn_3spt.pth               # CNN-3SPT model (22.4 MB)
│   ├── sag_vit.pth                # SAG-ViT model (25.9 MB)
│   ├── vit_pretrained_best.pth    # ViT model (327.5 MB)
│   └── resnet_cnn_model.pth       # ResNet50 model (42.8 MB)
├── TSR_Models/                     # Main model implementations
│   ├── generate_adv_dataset.py    # ✅ Generate adversarial datasets
│   ├── benchmark_and_eval.py      # Benchmark pre-trained models
│   ├── train_and_eval.py          # Transfer learning training
│   ├── cnn_3spt.py                # CNN-3SPT architecture
│   ├── SAG_VIT.py                 # SAG-ViT architecture
│   ├── models_factory.py          # Model loader utilities
│   ├── lisa.py                    # LISA dataset loader
│   ├── custom_data.py             # Custom dataset utilities
│   ├── model_components.py        # Shared model components
│   └── graph_construction.py      # Graph construction for SAG-ViT
├── Adversarial_Training/           # Alternative adversarial training scripts
├── data/                           # LISA dataset (auto-downloaded)
├── adversarial_dataset/            # Generated adversarial dataset
│   ├── train/                     # Training images (original)
│   ├── test/                      # Test images (with adversarial patches)
│   └── labels.json                # Label mappings
├── test_workflow.py                # ✅ Workflow verification script
├── test_models.py                  # ✅ Model benchmark test
└── README_WORKFLOW.md              # This file
```

## 🚀 Quick Start

### 1. Activate Environment

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Verify Installation

```powershell
python test_workflow.py
```

**Expected Output:**
- ✓ All dependencies installed
- ✓ LISA dataset loaded: 6621 training samples
- ✓ Adversarial patch generation works
- ✓ Model weights: 4/4 found
- ✓ Model architectures: Loadable

### 3. Test Models

```powershell
python test_models.py
```

**Expected Output:**
- ✓ CNN-3SPT Model: 5,857,818 parameters, ~55 images/sec
- ✓ SAG-ViT Model: 6,686,394 parameters, ~27 images/sec

## 📊 Complete Workflow

### Step 1: Generate Adversarial Dataset

```powershell
cd TSR_Models
python generate_adv_dataset.py
```

**What it does:**
- Loads LISA dataset (47 classes, 32x32 RGB images)
- Generates adversarial patches (occlusion, light, perturbation, localized)
- Creates train/test split
- Saves to `adversarial_dataset/`

**Output:**
- 50 original images in `train/`
- 200 adversarial images in `test/` (4 variants × 50 images)
- `labels.json` with class mappings

**Customization:**
```python
generate_adversarial_dataset(
    dataset, 
    num_samples=100,  # Increase for more data
    patch_types=['occlusion', 'light'],  # Select specific attacks
    size=30  # Patch size in pixels
)
```

### Step 2: Benchmark Pre-trained Models

```powershell
python benchmark_and_eval.py
```

**Requirements:**
- Data in `data/organized_images/` directory
- Pre-trained weights in `../Model_Weights/`

**What it evaluates:**
- ResNet50
- ViT-B/16
- CNN-3SPT
- SAG-ViT

**Outputs:**
- Accuracy metrics in `results/`
- Confusion matrices (PNG files)
- `eval_results.csv` with summary

### Step 3: Transfer Learning

```powershell
python train_and_eval.py
```

**What it does:**
- Loads pre-trained models
- Replaces final classification layers
- Fine-tunes on adversarial dataset
- Early stopping at 85% accuracy or 50 epochs

**Outputs:**
- Retrained models in `models/` (with `_retrained` suffix)
- Confusion matrices in `results/`
- `transfer_results.csv` with training history

## 🏗️ Model Architectures

### CNN-3SPT
- **Input:** 48×48 RGB images
- **Params:** 5.9M
- **Features:** 
  - Spatial Transformer Network (STN)
  - 3-stage processing
  - Adaptive pooling layers

### SAG-ViT (Spatial-Aware Graph Vision Transformer)
- **Input:** 224×224 RGB images
- **Params:** 6.7M
- **Features:**
  - Graph attention networks
  - Transformer encoder
  - Patch-based processing
  - EfficientNetV2 feature extraction

### ResNet50 & ViT
- **Input:** 224×224 RGB images
- **Backbone:** Pre-trained on ImageNet
- **Fine-tuned** for traffic sign recognition

## 🔧 Key Dependencies

All pre-installed in `.venv`:
- Python 3.13.1
- PyTorch + torchvision
- torch-geometric (for SAG-ViT)
- timm (model architectures)
- scikit-learn
- pandas, matplotlib
- opencv-python
- numpy

## 🎯 For Collaborators: Adding New Features

### Quick Start for Contributors

1. **Fork and clone** the repository
2. **Set up development environment** - see [CONTRIBUTING.md](CONTRIBUTING.md)
3. **Create a feature branch**: `git checkout -b feature/amazing-feature`
4. **Make your changes** following our [style guide](CONTRIBUTING.md#style-guidelines)
5. **Test thoroughly**: `python test_workflow.py && python test_models.py`
6. **Submit a pull request** with clear description

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Adding New Adversarial Patch Types

Edit `TSR_Models/generate_adv_dataset.py`:

```python
def apply_adversarial_patch(image, patch_type='occlusion', size=20):
    """
    Applies an adversarial modification to the image.
    
    Args:
        image: Torch tensor image (C, H, W)
        patch_type: Type of adversarial attack
        size: Size of the patch in pixels
    
    Returns:
        Modified image tensor
    """
    # ... existing code ...
    
    elif patch_type == 'your_new_attack':
        # Your implementation here
        # Example: Weather effects (fog, rain, etc.)
        image_modified = your_attack_function(image, size)
        
    # Always return properly formatted tensor
    return torch.tensor(image_modified.transpose((2, 0, 1)))
```

**Testing your new attack:**

```python
if __name__ == "__main__":
    from lisa import LISA
    import matplotlib.pyplot as plt
    
    # Load sample
    dataset = LISA(root='./data', train=True, download=True)
    image, label = dataset[0]
    
    # Test your attack
    attacked = apply_adversarial_patch(image, 'your_new_attack', size=20)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[1].imshow(attacked.permute(1, 2, 0))
    axes[1].set_title('Your Attack')
    plt.savefig('attack_comparison.png')
    print("✓ Visualization saved to attack_comparison.png")
```

Then update the patch_types list when generating:

```python
generate_adversarial_dataset(
    dataset,
    num_samples=100,
    patch_types=['occlusion', 'light', 'perturbation', 'your_new_attack'],
    size=20
)
```

### Adding New Model Architectures

**Step 1:** Create your model file `TSR_Models/your_model.py`:

```python
"""
Your Model Architecture for Traffic Sign Recognition.

This module implements [brief description of your model].
"""
import torch
import torch.nn as nn

class YourModelClass(nn.Module):
    """
    Your custom TSR model.
    
    Args:
        num_classes (int): Number of traffic sign classes. Default: 47 (LISA).
        input_size (int): Expected input image size. Default: 224.
        dropout (float): Dropout rate for regularization. Default: 0.5.
    
    Attributes:
        features: Feature extraction layers
        classifier: Classification head
    
    Example:
        >>> model = YourModelClass(num_classes=47)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = model(input_tensor)
        >>> output.shape
        torch.Size([1, 47])
    """
    def __init__(self, num_classes=47, input_size=224, dropout=0.5):
        super(YourModelClass, self).__init__()
        
        # Define your architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # ... add more layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

**Step 2:** Add loader to `TSR_Models/models_factory.py`:

```python
def load_your_model(num_classes, device, pretrained_path=None):
    """
    Load your custom model with optional pre-trained weights.
    
    Args:
        num_classes (int): Number of output classes
        device: torch.device for model placement
        pretrained_path (str, optional): Path to pre-trained weights
    
    Returns:
        torch.nn.Module: Loaded model on specified device
    """
    from your_model import YourModelClass
    
    model = YourModelClass(num_classes=num_classes)
    
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded pre-trained weights from {pretrained_path}")
    else:
        print("⚠ No pre-trained weights loaded, using random initialization")
    
    return model.to(device)
```

**Step 3:** Update benchmark script to include your model:

```python
# In benchmark_and_eval.py or test_models.py
from models_factory import load_your_model

# Add to model list
models = {
    'YourModel': load_your_model(num_classes=47, device=device)
}
```

**Step 4:** Document your model in README.md

### Modifying Training Parameters

Edit `TSR_Models/train_and_eval.py`:

```python
# ============= Training Configuration =============
# Adjust these parameters based on your needs

# Learning rate (typical range: 1e-5 to 1e-3)
LEARNING_RATE = 1e-4

# Early stopping criteria
TARGET_ACCURACY = 0.85  # Stop when validation accuracy reaches this
MAX_EPOCHS = 50         # Maximum training epochs

# Batch size (adjust based on available memory)
BATCH_SIZE = 32  # Reduce if out of memory, increase for faster training

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# Data loaders
train_loader = DataLoader(
    train_set, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4  # Adjust based on CPU cores
)
```

**Advanced training configurations:**

```python
# Using different optimizers
from torch.optim import SGD, AdamW

# SGD with momentum
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# AdamW (Adam with weight decay)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Learning rate schedulers
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Step decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
```

## 🐛 Troubleshooting

### Issue: "LISA dataset not found"
**Solution:** The dataset will auto-download (26.4 MB). Ensure internet connection.

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use CPU:
```python
DEVICE = torch.device("cpu")  # Force CPU usage
```

### Issue: Import errors
**Solution:** Ensure virtual environment is activated:
```powershell
.venv\Scripts\Activate.ps1
```

### Issue: FileNotFoundError for model weights
**Solution:** Check `Model_Weights/` directory contains:
- cnn_3spt.pth
- sag_vit.pth
- vit_pretrained_best.pth
- resnet_cnn_model.pth

## 📝 Code Quality Notes

All test/example code has been wrapped in `if __name__ == "__main__":` blocks to prevent execution during imports. This ensures:
- Clean module imports
- No unintended side effects
- Proper multiprocessing support on Windows

## 📈 Performance Benchmarks

**Hardware:** CPU (adjust for your GPU)

| Model | Parameters | Throughput | Input Size |
|-------|-----------|------------|------------|
| CNN-3SPT | 5.9M | ~55 img/s | 48×48 |
| SAG-ViT | 6.7M | ~27 img/s | 224×224 |
| ResNet50 | ~23M | ~40 img/s | 224×224 |
| ViT-B/16 | ~86M | ~20 img/s | 224×224 |

## 📚 Dataset Information

**LISA Dataset:**
- Classes: 47 (US traffic signs)
- Training samples: 6,621
- Test samples: ~1,300
- Image size: 32×32 RGB
- Source: Laboratory for Intelligent & Safe Automobiles

**Adversarial Variants:**
- **Occlusion:** Random colored patch
- **Light:** Bright reflection simulation
- **Perturbation:** Gaussian noise
- **Localized Perturbation:** Combined light + noise

## ✅ Verification Checklist

- [x] Python environment configured
- [x] All dependencies installed
- [x] LISA dataset downloaded
- [x] Adversarial dataset generated (251 files)
- [x] Model weights verified (4/4 present)
- [x] Models load successfully
- [x] Inference runs without errors
- [x] Documentation complete

## 🤝 Contributing

When extending this workflow:
1. Test your changes with `test_workflow.py`
2. Verify models load with `test_models.py`
3. Document new features in this README
4. Keep example code in `if __name__ == "__main__"` blocks

## 📧 Support

For questions about the workflow, check:
1. This README
2. Code comments in each module
3. Test scripts (`test_workflow.py`, `test_models.py`)

---

**Status:** ✅ Fully Functional (Verified February 8, 2026)  
**Ready for:** Feature additions, model improvements, experimentation
