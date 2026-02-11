import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import re
from PIL import Image

class TrafficSignDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.attack_types = []

        attack_types_list = ["light", "occlusion", "perturbation", "localized_perturbation"]

        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    file_path = os.path.join(root, file_name)

                    label_match = re.search(r'_label_(\d+)', file_name)
                    attack_match = next((attack for attack in attack_types_list if attack in file_name), "original")

                    if label_match:
                        label = int(label_match.group(1))
                        self.image_paths.append(file_path)
                        self.labels.append(label)
                        self.attack_types.append(attack_match)

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"❌ No images found in dataset at '{dataset_path}'.")

        print(f"✅ Successfully loaded dataset! Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # ✅ Load as PIL Image
        label = self.labels[idx]
        attack_type = self.attack_types[idx]

        if self.transform and isinstance(img, Image.Image):  # ✅ Apply transform only if it's PIL
            img = self.transform(img)

        return img, label, attack_type

def visualize_samples(dataset, samples_per_class=3):
    samples_dict = {}

    for img, label, attack in dataset:
        key = f"{label} - {attack}"
        if key not in samples_dict:
            samples_dict[key] = []
        if len(samples_dict[key]) < samples_per_class:
            samples_dict[key].append((img, key))
        if all(len(samples) >= samples_per_class for samples in samples_dict.values()):
            break

    num_classes = len(samples_dict)
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(9, num_classes * 3))

    for i, (key, images) in enumerate(samples_dict.items()):
        for j in range(samples_per_class):
            img, label = images[j]
            img = img.permute(1, 2, 0).cpu().numpy()  # ✅ Convert to NumPy
            img = (img - img.min()) / (img.max() - img.min())  # ✅ Normalize for display
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            axes[i, j].set_title(label)

    plt.tight_layout()
    plt.show()

# 🔹 Example Usage
DATASET_PATH = "adversarial_dataset"

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = TrafficSignDataset(dataset_path=DATASET_PATH, transform=transform)
test_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
visualize_samples(dataset)
