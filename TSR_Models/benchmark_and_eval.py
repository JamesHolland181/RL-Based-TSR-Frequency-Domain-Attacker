# eval_only.py
import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

from models_factory import *

# ---- Config ----
DATA_DIR = "data/organized_images"
BATCH_SIZE = 16
VALID_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- Dataset and Loaders ----
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
val_size = int(len(dataset) * VALID_SPLIT)
train_size = len(dataset) - val_size
_, val_set = random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

num_classes = len(dataset.classes)

# ---- Evaluation ----
models_to_eval = [
    ("resnet_cnn", lambda: load_resnet50(num_classes)),
    ("vit_pretrained_best", lambda: load_vit(num_classes)),
    ("cnn_3spt", lambda: load_cnn_3spt_arch(num_classes, DEVICE)),
    ("sag_vit", lambda: load_sag_vit_arch(num_classes, DEVICE)),
]

os.makedirs("results/", exist_ok=True)

eval_results = []

for model_name, model_loader in models_to_eval:
    print(f"\n🔍 Evaluating {model_name}...")
    model = model_loader().to(DEVICE)

    model_path = f"models/{model_name}.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model weights not found at {model_path}, skipping...")
        continue

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    duration = time.time() - start_time

    print(f"✅ {model_name} Accuracy: {acc:.4f} | Eval Time: {duration:.2f}s")

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    cm_path = f"results/{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Saved confusion matrix to {cm_path}")

    eval_results.append({
        "model": model_name,
        "accuracy": acc,
        "eval_time_sec": duration
    })

# ---- Save Summary ----
df = pd.DataFrame(eval_results)
print("\n📊 Evaluation Summary:")
print(df.to_string(index=False))
df.to_csv("eval_results.csv", index=False)
