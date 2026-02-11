import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image  # ✅ Import PIL for conversion
import random

class TrafficSignDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Custom dataset loader for the adversarial dataset structure.
        :param root: Path to dataset (should contain 'original', 'occlusion', etc.)
        :param transform: Optional transformations for the images.
        """
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 🔹 Iterate over dataset subsets (original, occlusion, etc.)
        for subset in os.listdir(root):
            subset_path = os.path.join(root, subset)
            labels_file = os.path.join(subset_path, "labels.json")

            if not os.path.isdir(subset_path) or not os.path.exists(labels_file):
                print(f"⚠ Skipping {subset_path} (No labels.json found)")
                continue

            # 🔹 Load labels
            with open(labels_file, "r") as f:
                labels_dict = json.load(f)

            # 🔹 Collect valid image paths and labels
            for filename, label in labels_dict.items():
                image_path = os.path.join(subset_path, filename)

                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                    self.labels.append(label)
                else:
                    print(f"⚠ Missing image {image_path}, skipping.")

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"❌ No images found in dataset at '{root}'. Check structure.")

        print(f"✅ Successfully loaded dataset! Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Loads an image and its label."""
        img_path = self.image_paths[idx]
        img = read_image(img_path).permute(1, 2, 0).numpy()  # Convert to HxWxC NumPy
        label = self.labels[idx]

        # ✅ Convert NumPy array to PIL Image
        img = Image.fromarray((img * 255).astype(np.uint8))  

        # ✅ Apply transformations correctly
        if self.transform:
            img = self.transform(img)

        return img, label


# 🔹 Set dataset path
DATASET_PATH = os.path.abspath("adversarial_dataset")  # Ensure path is correct

# 🔹 Check if dataset path exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: Dataset directory '{DATASET_PATH}' not found!")
    exit()

# 🔹 Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Function to Denormalize and Visualize Random Images
def denormalize(img_tensor, mean, std):
    """Reverses normalization for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img_tensor = img_tensor * std + mean  # Denormalize
    img_tensor = torch.clamp(img_tensor, 0, 1)  # Clip values to valid range
    return img_tensor

def visualize_random_samples(dataset, num_samples=5):
    """Displays a few random images from the dataset with labels."""
    indices = random.sample(range(len(dataset)), num_samples)  # Pick random indices
    samples = [dataset[i] for i in indices]  # Retrieve samples

    plt.figure(figsize=(15, 5))
    for i, (img_tensor, label) in enumerate(samples):
        img_tensor = denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_np = img_tensor.permute(1, 2, 0).numpy()  # Convert to HxWxC for plotting
        
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_np)
        plt.title(f"Label: {label}", fontsize=10)
        plt.axis("off")

    plt.show()

# Example usage
if __name__ == "__main__":
    # 🔹 Define transformations
    transform = transforms.Compose([
        transforms.Resize((48, 48)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 🔹 Load dataset
    test_dataset = TrafficSignDataset(root=DATASET_PATH, transform=transform)

    # 🔹 Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 🔹 Run Visualization
    visualize_random_samples(test_dataset, num_samples=5)
