# TSR Workflow Package - Setup Instructions

## 📦 Package Contents

**File:** `TSR_Workflow_Package_2026-02-08.zip` (568 MB)

This package contains a complete, tested, and ready-to-use Traffic Sign Recognition workflow with adversarial training capabilities.

### Included Files:

```
TSR_Workflow_Complete/
├── Model_Weights/              # Pre-trained model weights (418 MB)
│   ├── cnn_3spt.pth           # CNN-3SPT model
│   ├── sag_vit.pth            # SAG-ViT model  
│   ├── vit_pretrained_best.pth
│   └── resnet_cnn_model.pth
├── TSR_Models/                 # Model implementations
│   ├── generate_adv_dataset.py # Generate adversarial datasets ✅
│   ├── benchmark_and_eval.py   # Benchmark models
│   ├── train_and_eval.py       # Transfer learning
│   ├── cnn_3spt.py             # CNN-3SPT architecture
│   ├── SAG_VIT.py              # SAG-ViT architecture
│   ├── models_factory.py       # Model loaders
│   ├── lisa.py                 # LISA dataset loader
│   ├── custom_data.py          # Custom datasets
│   ├── model_components.py     # Shared components
│   └── graph_construction.py   # Graph construction
├── Adversarial_Training/       # Additional training scripts
├── adversarial_dataset/        # Pre-generated adversarial data (251 files)
│   ├── train/                  # Original training images
│   ├── test/                   # Adversarial test images
│   └── labels.json             # Label mappings
├── README.txt                  # Original readme
├── README_WORKFLOW.md          # Complete workflow documentation ⭐
├── test_workflow.py            # Quick verification script ✅
└── test_models.py              # Model benchmark test ✅
```

## 🚀 Quick Start (5 Minutes)

### Step 1: Extract the Package

Extract `TSR_Workflow_Package_2026-02-08.zip` to your preferred location.

### Step 2: Create Python Environment

**Windows (PowerShell):**
```powershell
cd TSR_Workflow_Complete
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
cd TSR_Workflow_Complete
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install torch torchvision torch-geometric scikit-learn pandas matplotlib opencv-python numpy timm
```

**Expected install time:** 2-5 minutes

### Step 4: Verify Installation

```bash
python test_workflow.py
```

**Expected output:**
```
============================================================
TSR WORKFLOW VERIFICATION
============================================================

[1/5] Checking dependencies...
✓ All dependencies installed

[2/5] Testing LISA dataset...
✓ LISA dataset loaded: 6621 training samples
  - Classes: 47
  - Image shape: torch.Size([3, 32, 32])

[3/5] Testing adversarial patch generation...
✓ Adversarial patch generation works
  - Original shape: torch.Size([3, 32, 32])
  - Modified shape: torch.Size([3, 32, 32])

[4/5] Checking pre-trained model weights...
  ✓ cnn_3spt.pth (22.4 MB)
  ✓ sag_vit.pth (25.9 MB)
  ✓ vit_pretrained_best.pth (327.5 MB)
  ✓ resnet_cnn_model.pth (42.8 MB)

[5/5] Testing model architectures...
✓ CNN-3SPT architecture
✓ SAG-ViT architecture

============================================================
WORKFLOW READY TO USE
============================================================
```

### Step 5: Test Models (Optional)

```bash
python test_models.py
```

This runs inference on the pre-generated adversarial dataset and verifies models load correctly.

## 🔥 Using the Workflow

### Generate More Adversarial Data

```bash
cd TSR_Models
python generate_adv_dataset.py
```

**Customize in the script:**
```python
generate_adversarial_dataset(
    dataset, 
    num_samples=100,  # Number of base images
    patch_types=['occlusion', 'light', 'perturbation', 'localized_perturbation'],
    size=20  # Patch size in pixels
)
```

### Benchmark Pre-trained Models

```bash
python benchmark_and_eval.py
```

**Requirements:** Organized image dataset in `data/organized_images/`

### Fine-tune with Transfer Learning

```bash
python train_and_eval.py
```

**What it does:**
- Loads pre-trained models
- Replaces final layers
- Trains on adversarial dataset
- Early stopping at 85% accuracy

## 📖 Complete Documentation

See **README_WORKFLOW.md** for:
- Detailed architecture descriptions
- Adding new adversarial attacks
- Adding new model architectures
- Troubleshooting guide
- Performance benchmarks
- Contributing guidelines

## 🎯 Key Features Verified

✅ **LISA Dataset:** Auto-downloads (26.4 MB), 47 traffic sign classes  
✅ **Adversarial Generation:** 4 attack types (occlusion, light, perturbation, localized)  
✅ **Pre-trained Models:** 4 models ready to use (CNN-3SPT, SAG-ViT, ResNet50, ViT)  
✅ **Transfer Learning:** Fine-tuning pipeline with early stopping  
✅ **Evaluation:** Confusion matrices, accuracy metrics, visualization  

## 💡 What's NOT Included

The following are excluded to keep package size reasonable (they download/generate automatically):

- ❌ Python virtual environment (.venv) - **Create with Step 2**
- ❌ Raw LISA dataset (data/) - **Auto-downloads on first run**
- ❌ Python package cache (__pycache__) - **Regenerates automatically**

## ⚙️ System Requirements

**Minimum:**
- Python 3.8+
- 4 GB RAM
- 2 GB disk space

**Recommended:**
- Python 3.10+
- 8 GB RAM  
- CUDA-capable GPU (optional, speeds up training)
- 5 GB disk space

## 🐛 Common Issues

### Issue: "ModuleNotFoundError"
**Solution:** Ensure you activated the virtual environment and installed all dependencies:
```bash
.venv\Scripts\Activate.ps1  # Windows
pip install torch torchvision torch-geometric scikit-learn pandas matplotlib opencv-python numpy timm
```

### Issue: "CUDA out of memory"
**Solution:** The code automatically falls back to CPU. To force CPU:
```python
DEVICE = torch.device("cpu")
```

### Issue: Slow on CPU
**Solution:** Reduce batch size in training scripts:
```python
train_loader = DataLoader(train_set, batch_size=16)  # Default is 64
```

## 📊 Performance Expectations

**On CPU (typical laptop):**
- Adversarial generation: ~2 min for 50 samples
- CNN-3SPT inference: ~55 images/sec
- SAG-ViT inference: ~27 images/sec
- Training epoch: 5-10 minutes

**On GPU (CUDA):**
- 5-10x faster inference and training

## ✉️ Questions?

Refer to:
1. **README_WORKFLOW.md** - Complete technical documentation
2. Code comments in each script
3. `test_workflow.py` and `test_models.py` for working examples

## 🎉 Ready to Go!

The workflow has been tested and verified working as of February 8, 2026. All components are functional and ready for feature development, experimentation, and research.

**Happy coding! 🚀**

---

**Package Created:** February 8, 2026  
**Verification Status:** ✅ All components tested and working  
**Python Version:** 3.8+ (tested on 3.13.1)
