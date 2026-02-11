import torch
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
from lisa import LISA

def load_lisa_dataset(root='path_to_dataset', train=True, download=True):
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
        x, y = random.randint(0, w - size), random.randint(0, h - size)
        image[y:y+size, x:x+size] = np.random.rand(3)  # Random RGB occlusion

    elif patch_type == 'light':
        mask = np.zeros((h, w, c), dtype=np.float32)
        center = (random.randint(size, w - size), random.randint(size, h - size))
        cv2.circle(mask, center, size, (1.0, 1.0, 1.0), -1)
        image = np.clip(image + 0.3 * mask, 0, 1)

    elif patch_type == 'perturbation':
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image + noise, 0, 1)

    elif patch_type == 'localized_perturbation':
        mask = np.zeros((h, w, c), dtype=np.float32)
        anchor_count = random.randint(3, 6)
        pts = np.array([[random.randint(0, w - 1), random.randint(0, h - 1)] for _ in range(anchor_count)], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (1.0, 1.0, 1.0))
        noise = np.random.normal(0, 0.1, image.shape) * mask
        image = np.clip(image + 0.3 * mask + noise, 0, 1)

    return torch.tensor(image.transpose((2, 0, 1)))  # Convert back to CxHxW

def generate_adversarial_dataset(dataset,
                                 patch_types=['occlusion', 'light', 'perturbation', 'localized_perturbation'],
                                 size=20,
                                 save_dir='adversarial_dataset'):
    """
    Generates an adversarial dataset, grouped by traffic sign type.
    
    :param dataset: The original dataset to modify.
    :param patch_types: Types of adversarial attacks.
    :param size: Size of the patch in pixels.
    :param save_dir: Directory to save the adversarial dataset.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Dictionary to store which samples have already been selected per class
    selected_samples = {}

    for index in range(len(dataset)):
        image, label = dataset[index]

        # Ensure each class has only one sample per attack
        if label not in selected_samples:
            selected_samples[label] = []

        # Skip if we already have all attacks for this class
        if len(selected_samples[label]) >= len(patch_types):
            continue

        # Create class directory
        class_dir = os.path.join(save_dir, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)

        for patch_type in patch_types:
            if patch_type in selected_samples[label]:
                continue  # Avoid duplicates

            adversarial_image = apply_adversarial_patch(image.clone(), patch_type, size)
            adversarial_filename = f'image_{index}_label_{label}_{patch_type}.png'
            adversarial_image_path = os.path.join(class_dir, adversarial_filename)

            # Save adversarial image
            plt.imsave(adversarial_image_path, adversarial_image.numpy().transpose((1, 2, 0)))

            # Mark attack type as used for this class
            selected_samples[label].append(patch_type)

            # Stop if we have all attacks for this class
            if len(selected_samples[label]) == len(patch_types):
                break

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
    dataset = load_lisa_dataset()
    generate_adversarial_dataset(dataset)
    visualize_adversarial_sample(dataset, index=0, patch_type='localized_perturbation', size=30)
