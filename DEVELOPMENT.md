## Development Setup

### For Contributors

If you're planning to contribute to the project, follow these additional setup steps:

#### 1. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available
```

For code quality tools:

```bash
pip install black flake8 pytest pytest-cov
```

#### 2. Configure Git Hooks (Optional)

Set up pre-commit hooks to automatically check code quality:

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']
```

#### 3. IDE Setup

**VS Code (Recommended)**

Install extensions:
- Python (Microsoft)
- Pylance
- Jupyter
- GitLens

Create `.vscode/settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

**PyCharm**

1. Open Settings → Project → Python Interpreter
2. Select the virtual environment (`.venv`)
3. Enable "Black" formatter
4. Configure "flake8" as linter

#### 4. Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=TSR_Models --cov-report=html

# Run specific test file
pytest test_workflow.py

# Run with verbose output
pytest -v
```

#### 5. Building Documentation (If using Sphinx)

```bash
cd docs
make html
```

View documentation at `docs/_build/html/index.html`

## Environment Variables

You can configure the following environment variables:

```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Set default dataset path
export TSR_DATASET_PATH="./custom_data"

# Set model weights directory
export TSR_MODEL_WEIGHTS="./my_weights"
```

## Performance Optimization

### GPU Acceleration

Ensure CUDA is properly installed:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Install CUDA-specific PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Memory Management

For large datasets or models:

```python
# Enable automatic mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### Multi-GPU Training

```python
import torch.nn as nn

# Wrap model for data parallelism
model = nn.DataParallel(model)
```

## Customization Guide

### Custom Dataset Integration

Create a custom dataset class:

```python
from torch.utils.data import Dataset
import os
from PIL import Image

class CustomTSRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load your data
        for class_idx, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            for img_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, img_name))
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### Custom Model Integration

Add your model to the framework:

```python
# TSR_Models/my_custom_model.py
import torch.nn as nn

class MyCustomTSR(nn.Module):
    """
    Your custom traffic sign recognition model.
    
    Args:
        num_classes (int): Number of traffic sign classes
        input_size (int): Input image size (assumes square images)
    """
    def __init__(self, num_classes=47, input_size=224):
        super(MyCustomTSR, self).__init__()
        
        # Define your architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ... more layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

Register in `models_factory.py`:

```python
def load_my_custom_model(num_classes, device, pretrained_path=None):
    """Load custom model with optional pre-trained weights."""
    from my_custom_model import MyCustomTSR
    
    model = MyCustomTSR(num_classes=num_classes)
    
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f"✓ Loaded pre-trained weights from {pretrained_path}")
    
    return model.to(device)
```

## Deployment

### Model Export for Production

#### Export to ONNX

```python
import torch
import torch.onnx

# Load your trained model
model = load_cnn_3spt(num_classes=47, device='cpu')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 48, 48)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "cnn_3spt.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

#### TorchScript Export

```python
# Trace the model
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("cnn_3spt_traced.pt")

# Or use scripting
scripted_model = torch.jit.script(model)
scripted_model.save("cnn_3spt_scripted.pt")
```

#### Quantization for Edge Devices

```python
import torch.quantization

# Post-training static quantization
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate with representative data
with torch.no_grad():
    for data, _ in calibration_loader:
        model(data)

# Convert to quantized model
quantized_model = torch.quantization.convert(model, inplace=True)

# Save quantized model
torch.save(quantized_model.state_dict(), 'model_quantized.pth')
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for API (if applicable)
EXPOSE 8000

# Run application
CMD ["python", "serve_model.py"]
```

Build and run:

```bash
docker build -t tsr-model .
docker run -p 8000:8000 tsr-model
```

## Common Development Tasks

### Adding a New Attack Type

1. Edit `TSR_Models/generate_adv_dataset.py`
2. Add your attack in `apply_adversarial_patch()`
3. Test with sample images
4. Update documentation
5. Submit PR

### Retraining Models

```bash
# Full training from scratch
python TSR_Models/train_and_eval.py --epochs 100 --lr 0.001

# Transfer learning
python TSR_Models/train_and_eval.py --pretrained --freeze-backbone

# Resume from checkpoint
python TSR_Models/train_and_eval.py --resume checkpoints/epoch_50.pth
```

### Benchmarking New Models

Add to `test_models.py`:

```python
def test_your_model():
    from models_factory import load_your_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_your_model(num_classes=47, device=device)
    
    # Test loading
    assert model is not None
    
    # Test inference
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    assert output.shape == (1, 47)
    
    print("✓ Your model test passed")
```

## Troubleshooting Development Issues

### Import Errors in Development

If you encounter import errors when developing:

```python
# Add to top of your script
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

### Memory Leaks During Training

```python
# Clear cache regularly
import gc
torch.cuda.empty_cache()
gc.collect()
```

### Debugging Tips

Enable anomaly detection:

```python
torch.autograd.set_detect_anomaly(True)
```

Print model structure:

```python
from torchinfo import summary
summary(model, input_size=(1, 3, 224, 224))
```

---

For more help, see [CONTRIBUTING.md](CONTRIBUTING.md) or open an issue.
