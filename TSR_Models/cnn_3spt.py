# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import timm
from lisa import LISA 
import matplotlib.pyplot as plt
import os

from custom_data import TrafficSignDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# train_dataset = LISA(root='./data', download=True, train=True, transform=transform)
# test_dataset  = LISA(root='./data', download=False, train=False, transform=transform)

# test_dataset = TrafficSignDataset("adversarial_dataset")

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
# test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)

# %%
num_classes = 47

# %%
class STN(nn.Module):
    def __init__(self, in_channels, input_size, 
                 kernel_size1=7, kernel_size2=5, 
                 padding1=0, padding2=0, 
                 output_size=(4,4)):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=kernel_size1, padding=padding1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=kernel_size2, padding=padding2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size)
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * output_size[0] * output_size[1], 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
    
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def forward(self, x):
        xs = self.localization(x)
        xs = self.adaptive_pool(xs)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

# %%
# Full architecture with three STN modules 
class TrafficSignCNNFull(nn.Module):
    def __init__(self, num_classes, use_stn=True):
        super(TrafficSignCNNFull, self).__init__()
        self.use_stn = use_stn
        # STN1 operates on the raw input (3 channels, 48x48)
        if self.use_stn:
            self.stn1 = STN(in_channels=3, input_size=48,
                            kernel_size1=7, kernel_size2=5, padding1=0, padding2=0, output_size=(4,4))
        # Convolutional Block 1: approximates the first block from the paper
        self.conv1 = nn.Conv2d(3, 200, kernel_size=7, padding=2)  # output 46x46
        self.bn1 = nn.BatchNorm2d(200)
        self.pool = nn.MaxPool2d(2, 2)   # reduces spatial dims by half
        # STN2 on output of Block 1 (200 channels, 23x23)
        if self.use_stn:
            self.stn2 = STN(in_channels=200, input_size=23,
                            kernel_size1=7, kernel_size2=5, padding1=0, padding2=0, output_size=(4,4))
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(200, 250, kernel_size=4, padding=1)  # output: 22x22 -> pool -> 11x11
        self.bn2 = nn.BatchNorm2d(250)
        # STN3 on output of Block 2 (250 channels, 11x11)
        if self.use_stn:
            self.stn3 = STN(in_channels=250, input_size=11,
                            kernel_size1=3, kernel_size2=3, padding1=1, padding2=1, output_size=(4,4))
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(250, 350, kernel_size=4, padding=1)  # output: (11+2-4)+1 = 10 => 10x10
        self.bn3 = nn.BatchNorm2d(350)
        # Pooling: 10x10 -> floor(10/2)=5, so final feature map: 350 x 5 x 5
        # Fully connected layers: 350*5*5 = 8750
        self.fc1 = nn.Linear(350 * 5 * 5, 400)
        self.fc2 = nn.Linear(400, num_classes)
        
    def forward(self, x):
        # Apply STN1 on the input image
        if self.use_stn:
            x = self.stn1(x)
        # Convolution Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # output: 200 x 23 x 23
        # Apply STN2 on the features after Block 1
        if self.use_stn:
            x = self.stn2(x)
        # Convolution Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # output: 250 x 11 x 11
        # Apply STN3 on the features after Block 2
        if self.use_stn:
            x = self.stn3(x)
        # Convolution Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)  # output: 350 x 5 x 5
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage - for testing the model directly
if __name__ == "__main__":
    # %%
    loaded_model = TrafficSignCNNFull(num_classes=num_classes, use_stn=True).to(device)
    loaded_model.load_state_dict(torch.load("cnn_3spt.pth", map_location = "cpu"))

    # %%
    loaded_model.eval()
    correct = 0
    total = 0
    img_paths = ""
    with torch.no_grad():
        for images, labels, img_paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = loaded_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # %%
    import numpy as np
    import matplotlib.pyplot as plt
    import torch, random

    loaded_model.eval()

    indices_by_class = {i: [] for i in range(len(test_dataset.classes))}
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        # Convert tensor label to int if necessary
        label = label.item() if isinstance(label, torch.Tensor) else label
        indices_by_class[label].append(idx)

    random_indices = []
    for class_idx in range(len(test_dataset.classes)):
        if indices_by_class[class_idx]:
            random_indices.append(random.choice(indices_by_class[class_idx]))

    selected_images = []
    true_labels = []
    for idx in random_indices:
        img, label = test_dataset[idx]
        selected_images.append(img.unsqueeze(0))
        true_labels.append(label.item() if isinstance(label, torch.Tensor) else label)

    images_tensor = torch.cat(selected_images, dim=0).to(device)

    with torch.no_grad():
        outputs = loaded_model(images_tensor)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    images_np = images_tensor.cpu().numpy()
    images_np = images_np * 0.5 + 0.5  


    n_ = len(random_indices)
    ncols = 7  
    nrows = (n_images // ncols) + (1 if n_images % ncols else 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2 * nrows))
    axes = axes.flatten()

    for i in range(n_images):
        ax = axes[i]
        ax.imshow(np.transpose(images_np[i], (1, 2, 0)))
        ax.set_title(f"Pred: {test_dataset.classes[preds[i]]}\nTrue: {test_dataset.classes[true_labels[i]]}")
        ax.axis("off")

    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
