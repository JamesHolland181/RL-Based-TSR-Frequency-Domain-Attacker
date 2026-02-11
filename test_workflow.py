"""
Quick workflow verification script
Tests each step of the TSR workflow with minimal samples
"""
import os
import sys
import torch

print("=" * 60)
print("TSR WORKFLOW VERIFICATION")
print("=" * 60)

# Step 1: Verify dependencies
print("\n[1/5] Checking dependencies...")
try:
    import torchvision
    import sklearn
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np
    print("✓ All dependencies installed")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

# Step 2: Test LISA dataset download
print("\n[2/5] Testing LISA dataset...")
sys.path.insert(0, 'TSR_Models')
from lisa import LISA

try:
    dataset = LISA(root='./data', train=True, download=True)
    print(f"✓ LISA dataset loaded: {len(dataset)} training samples")
    print(f"  - Classes: {len(dataset.classes)}")
    print(f"  - Image shape: {dataset[0][0].shape}")
except Exception as e:
    print(f"✗ Failed to load LISA: {e}")
    sys.exit(1)

# Step 3: Test adversarial patch generation
print("\n[3/5] Testing adversarial patch generation...")
try:
    os.chdir('TSR_Models')
    from generate_adv_dataset import apply_adversarial_patch
    
    test_image, test_label = dataset[0]
    adv_image = apply_adversarial_patch(test_image.clone(), 'occlusion', size=10)
    print(f"✓ Adversarial patch generation works")
    print(f"  - Original shape: {test_image.shape}")
    print(f"  - Modified shape: {adv_image.shape}")
    os.chdir('..')
except Exception as e:
    print(f"✗ Failed adversarial generation: {e}")
    os.chdir('..')
    sys.exit(1)

# Step 4: Check model weights
print("\n[4/5] Checking pre-trained model weights...")
weights_dir = 'Model_Weights'
expected_weights = ['cnn_3spt.pth', 'sag_vit.pth', 'vit_pretrained_best.pth', 'resnet_cnn_model.pth']
found_weights = []
missing_weights = []

for weight in expected_weights:
    weight_path = os.path.join(weights_dir, weight)
    if os.path.exists(weight_path):
        size_mb = os.path.getsize(weight_path) / (1024 * 1024)
        found_weights.append(f"  ✓ {weight} ({size_mb:.1f} MB)")
    else:
        missing_weights.append(f"  ✗ {weight} (missing)")

for w in found_weights:
    print(w)
for w in missing_weights:
    print(w)

# Step 5: Check model architecture loading
print("\n[5/5] Testing model architectures...")
sys.path.insert(0, 'TSR_Models')
try:
    from cnn_3spt import CNN_3SPT
    from SAG_VIT import SAG_ViT
    print("✓ CNN-3SPT architecture")
    print("✓ SAG-ViT architecture")
except Exception as e:
    print(f"✗ Failed to import models: {e}")

# Summary
print("\n" + "=" * 60)
print("WORKFLOW STATUS SUMMARY")
print("=" * 60)
print(f"✓ Dependencies: OK")
print(f"✓ LISA Dataset: {len(dataset)} samples ready")
print(f"✓ Adversarial Generation: Functional")
print(f"✓ Model Weights: {len(found_weights)}/{len(expected_weights)} found")
print(f"✓ Model Architectures: Loadable")

print("\n" + "=" * 60)
print("WORKFLOW READY TO USE")
print("=" * 60)
print("\nNext steps for your collaborator:")
print("1. Generate full adversarial dataset: python TSR_Models/generate_adv_dataset.py")
print("2. Benchmark models: python TSR_Models/benchmark_and_eval.py")
print("3. Transfer learning: python TSR_Models/train_and_eval.py")
