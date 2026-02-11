import torch
import numpy as np
import cv2
import random
import os
import json
from lisa import LISA

def load_lisa_dataset(root='./data', train=True, download=True):
    """Loads the LISA dataset using the custom class from lisa.py."""
    dataset = LISA(root=root, train=train, download=download)
    return dataset

def apply_adversarial_patch(image, patch_type='occlusion', size=20):
    """
    Applies an adversarial modification to the image.
    
    :param image: Torch tensor image (C, H, W)
    :param patch_type: Type of adversarial attack (occlusion, light, perturbation, localized_perturbation)
    :param size: Size of the patch in pixels
    :return: Modified image tensor
    """
    image = image.numpy().transpose((1, 2, 0)).astype(np.float32)  # Convert from CxHxW to HxWxC
    h, w, c = image.shape
    max_size = int(0.25 * min(h, w))  # Ensure the disturbance is not larger than 25% of the image size
    size = min(size, max_size)
    
    if patch_type == 'occlusion':
        # Create a solid color patch (random location)
        x, y = random.randint(0, w - size), random.randint(0, h - size)
        image[y:y+size, x:x+size] = np.random.rand(3)  # Random RGB occlusion
    
    elif patch_type == 'light':
        # Simulate a bright reflection
        mask = np.zeros((h, w, c), dtype=np.float32)
        center = (random.randint(size, w - size), random.randint(size, h - size))
        cv2.circle(mask, center, size, (1.0, 1.0, 1.0), -1)
        image = np.clip(image + 0.3 * mask, 0, 1)  # Increase brightness
    
    elif patch_type == 'perturbation':
        # Add small noise to simulate adversarial perturbations
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image + noise, 0, 1)
    
    elif patch_type == 'localized_perturbation':
        # Apply localized perturbation combining light and noise effects
        mask = np.zeros((h, w, c), dtype=np.float32)
        center = (random.randint(size, w - size), random.randint(size, h - size))
        cv2.circle(mask, center, size, (1.0, 1.0, 1.0), -1)
        noise = np.random.normal(0, 0.1, image.shape) * mask  # Apply noise only inside the circle
        image = np.clip(image + 0.3 * mask + noise, 0, 1)
    
    return torch.tensor(image.transpose((2, 0, 1)))  # Convert back to CxHxW

def generate_adversarial_dataset(dataset, num_samples=50, patch_types=['occlusion', 'light', 'perturbation', 'localized_perturbation'], size=20, save_dir='adversarial_dataset'):
    """Generates an adversarial dataset in the LISA-style format."""
    train_dir = os.path.join(save_dir, "train")
    test_dir = os.path.join(save_dir, "test")
    labels_file = os.path.join(save_dir, "labels.json")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    labels_dict = {}  # Store label mappings

    for i in range(num_samples):
        if i % 10 == 0:
            print(f"Processing sample {i}/{num_samples}...")
        index = random.randint(0, len(dataset) - 1)
        image, label = dataset[index]
        
        # Create class subdirectories if they don't exist
        class_dir_train = os.path.join(train_dir, f"class_{int(label)}")
        class_dir_test = os.path.join(test_dir, f"class_{int(label)}")
        os.makedirs(class_dir_train, exist_ok=True)
        os.makedirs(class_dir_test, exist_ok=True)

        # Save original image using cv2 (faster)
        original_filename = f"image_{i}.png"
        original_image_path = os.path.join(class_dir_train, original_filename)
        img_np = (image.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        cv2.imwrite(original_image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        labels_dict[original_filename] = int(label)

        # Generate and save adversarial images
        for patch_type in patch_types:
            adversarial_image = apply_adversarial_patch(image.clone(), patch_type, size)
            adversarial_filename = f"image_{i}_{patch_type}.png"
            adversarial_image_path = os.path.join(class_dir_test, adversarial_filename)
            adv_np = (adversarial_image.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            cv2.imwrite(adversarial_image_path, cv2.cvtColor(adv_np, cv2.COLOR_RGB2BGR))
            
            labels_dict[adversarial_filename] = int(label)
    
    # Save labels as JSON
    with open(labels_file, "w") as f:
        json.dump(labels_dict, f, indent=4)

    print(f"✅ Adversarial dataset generated successfully at {save_dir}")

# Example usage
if __name__ == "__main__":
    print("Loading LISA dataset...")
    dataset = load_lisa_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")
    print("Generating adversarial dataset...")
    generate_adversarial_dataset(dataset, num_samples=50)
    print("Done!")
