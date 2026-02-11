import torch
import numpy as np
import matplotlib.pyplot as plt
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

def generate_adversarial_dataset(dataset, num_samples=200, patch_types=['occlusion', 'light', 'perturbation', 'localized_perturbation'], size=20, save_dir='adversarial_dataset'):
    """Generates an adversarial dataset in the LISA-style format."""
    train_dir = os.path.join(save_dir, "train")
    test_dir = os.path.join(save_dir, "test")
    labels_file = os.path.join(save_dir, "labels.json")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    labels_dict = {}  # Store label mappings

    for i in range(num_samples):
        index = random.randint(0, len(dataset) - 1)
        image, label = dataset[index]
        
        # Create class subdirectories if they don’t exist
        class_dir_train = os.path.join(train_dir, f"class_{label}")
        class_dir_test = os.path.join(test_dir, f"class_{label}")
        os.makedirs(class_dir_train, exist_ok=True)
        os.makedirs(class_dir_test, exist_ok=True)

        # Save original image
        original_filename = f"image_{i}.png"
        original_image_path = os.path.join(class_dir_train, original_filename)
        plt.imsave(original_image_path, image.numpy().transpose((1, 2, 0)))
        
        labels_dict[original_filename] = label  # Store label mapping

        # Generate and save adversarial images
        for patch_type in patch_types:
            adversarial_image = apply_adversarial_patch(image.clone(), patch_type, size)
            adversarial_filename = f"image_{i}_{patch_type}.png"
            adversarial_image_path = os.path.join(class_dir_test, adversarial_filename)
            plt.imsave(adversarial_image_path, adversarial_image.numpy().transpose((1, 2, 0)))
            
            labels_dict[adversarial_filename] = label  # Store label mapping
    
    # Save labels as JSON
    with open(labels_file, "w") as f:
        json.dump(labels_dict, f, indent=4)

    print(f"✅ Adversarial dataset generated successfully at {save_dir}")

def visualize_adversarial_sample(dataset, index=0, patch_type='occlusion', size=20):
    """Displays an original and adversarially modified image side by side."""
    image, label = dataset[index]
    adversarial_image = apply_adversarial_patch(image.clone(), patch_type, size)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image.numpy().transpose((1, 2, 0)))
    ax[0].set_title(f'Original (Label: {label})')
    ax[0].axis('off')
    
    ax[1].imshow(adversarial_image.numpy().transpose((1, 2, 0)))
    ax[1].set_title(f'Adversarial ({patch_type})')
    ax[1].axis('off')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    print("Loading LISA dataset...")
    dataset = load_lisa_dataset()
    print(f"Dataset loaded with {len(dataset)} samples")
    print("Generating adversarial dataset (50 samples for testing)...")
    generate_adversarial_dataset(dataset, num_samples=50)
    print("Done!")
    # visualize_adversarial_sample(dataset, index=0, patch_type='occlusion', size=30)
