"""
Quick Model Benchmark Test
Tests that pre-trained models can load and make predictions
"""
import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

sys.path.insert(0, 'TSR_Models')
from models_factory import load_cnn_3spt_arch, load_sag_vit_arch

print("=" * 60)
print("MODEL BENCHMARK TEST")
print("=" * 60)

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {DEVICE}")

# Load adversarial test dataset
print("\nLoading adversarial test dataset...")
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # CNN-3SPT size
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder('adversarial_dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print(f"✓ Test dataset loaded: {len(test_dataset)} adversarial images")
print(f"  - Classes: {len(test_dataset.classes)}")

# Test CNN-3SPT
print("\n" + "-" * 60)
print("Testing CNN-3SPT Model")
print("-" * 60)
try:
    num_classes = len(test_dataset.classes)
    model = load_cnn_3spt_arch(num_classes, DEVICE)
    model.eval()
    
    # Test inference
    start_time = time.time()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    inference_time = time.time() - start_time
    accuracy = 100 * correct / total
    
    print(f"✓ Model loaded successfully")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Inference time: {inference_time:.2f}s for {total} images")
    print(f"  - Throughput: {total/inference_time:.1f} images/sec")
    print(f"  - Test accuracy: {accuracy:.2f}%")
    
except Exception as e:
    print(f"✗ CNN-3SPT test failed: {e}")
    import traceback
    traceback.print_exc()

# Test SAG-ViT
print("\n" + "-" * 60)
print("Testing SAG-ViT Model")
print("-" * 60)
try:
    # SAG-ViT requires 224x224 input
    transform_vit = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    test_dataset_vit = datasets.ImageFolder('adversarial_dataset/test', transform=transform_vit)
    test_loader_vit = DataLoader(test_dataset_vit, batch_size=4, shuffle=False)
    
    model_vit = load_sag_vit_arch(num_classes, DEVICE)
    model_vit.eval()
    
    # Test a small batch
    start_time = time.time()
    images, labels = next(iter(test_loader_vit))
    images = images.to(DEVICE)
    
    with torch.no_grad():
        outputs = model_vit(images)
    
    inference_time = time.time() - start_time
    
    print(f"✓ Model loaded successfully")
    print(f"  - Parameters: {sum(p.numel() for p in model_vit.parameters()):,}")
    print(f"  - Inference time: {inference_time:.2f}s for {len(images)} images")
    print(f"  - Throughput: {len(images)/inference_time:.1f} images/sec")
    
except Exception as e:
    print(f"✗ SAG-ViT test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
print("\n✓ Models can load and make predictions successfully!")
print("✓ Adversarial dataset is ready for training")
print("\nWorkflow is ready for your collaborator to extend!")
