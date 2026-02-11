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

### Adding New Adversarial Patch Types

Edit `TSR_Models/generate_adv_dataset.py`:

```python
def apply_adversarial_patch(image, patch_type='occlusion', size=20):
    # ... existing code ...
    
    elif patch_type == 'your_new_attack':
        # Your implementation here
        image_modified = your_attack_function(image)
    
    return torch.tensor(image_modified.transpose((2, 0, 1)))
```

Then update the patch_types list when generating:

```python
generate_adversarial_dataset(
    dataset,
    patch_types=['occlusion', 'light', 'your_new_attack']
)
```

### Adding New Model Architectures

1. Create `TSR_Models/your_model.py` with your architecture
2. Add loader function to `TSR_Models/models_factory.py`:

```python
def load_your_model(num_classes, device):
    from your_model import YourModelClass
    model = YourModelClass(num_classes=num_classes)
    # Load pre-trained weights if available
    model.load_state_dict(torch.load('../Model_Weights/your_model.pth'))
    return model.to(device)
```

3. Update benchmark/training scripts to include your model

### Modifying Training Parameters

Edit `TSR_Models/train_and_eval.py`:

```python
# Learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adjust here

# Early stopping
TARGET_ACCURACY = 0.85  # Change target accuracy
MAX_EPOCHS = 50         # Change max epochs

# Batch size
train_loader = DataLoader(train_set, batch_size=32)  # Adjust batch size
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
