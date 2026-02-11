import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

from models_factory import (
    load_resnet50,
    load_vit,
    load_cnn_3spt_arch,
    load_sag_vit_arch
)

# ---- Config ----
DATA_DIR = "data/organized_images"
BATCH_SIZE = 32
VALID_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFER_LR = 1e-4
PRETRAINED_CLASS_COUNT = 671
TARGET_ACC = 0.95
EPOCH_LIMIT = 250

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
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

num_classes = len(dataset.classes)

# ---- Model Registry ----
models_to_eval = [
    ("resnet_cnn_retrained", lambda: load_resnet50(PRETRAINED_CLASS_COUNT)),
    ("vit_pretrained_best_retrained", lambda: load_vit(PRETRAINED_CLASS_COUNT)),
    ("cnn_3spt_retrained", lambda: load_cnn_3spt_arch(PRETRAINED_CLASS_COUNT, DEVICE)),
    ("sag_vit_retrained", lambda: load_sag_vit_arch(PRETRAINED_CLASS_COUNT, DEVICE)),
]

os.makedirs("results/", exist_ok=True)
os.makedirs("models/", exist_ok=True)

eval_results = []

for model_name, model_loader in models_to_eval:
    print(f"\n🔁 Transfer learning with {model_name}...")
    model = model_loader().to(DEVICE)

    model_path = f"models/{model_name}.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model weights not found at {model_path}, skipping...")
        continue

    # Load weights
    state_dict = torch.load(model_path, map_location=DEVICE)
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model.state_dict() and v.shape == model.state_dict()[k].shape
    }
    model.load_state_dict(filtered_state_dict, strict=False)

    # ---- Replace Final Layer ----
    if model_name == "resnet_cnn_retrained":
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(DEVICE)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_name == "vit_pretrained_best_retrained":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes).to(DEVICE)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.head.parameters():
            param.requires_grad = True

    elif model_name == "cnn_3spt_retrained":
        model.fc2 = nn.Linear(model.fc1.out_features, num_classes).to(DEVICE)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = True

    elif model_name == "sag_vit_retrained":
        # Dynamically get input dimension
        for layer in model.mlp.modules():
            if isinstance(layer, nn.Linear):
                in_dim = layer.in_features
                break
        else:
            raise ValueError("No Linear layer found in model.mlp")

        model.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_classes)
        ).to(DEVICE)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.mlp.parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # ---- Fine-tune Head with Early Stopping ----
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=TRANSFER_LR)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    epoch = 1
    start_time = time.time()

    while epoch <= EPOCH_LIMIT:
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"[{model_name}] Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}")

        # ---- Validation Step
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"[{model_name}] Validation Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc

        if acc >= TARGET_ACC:
            print(f"🛑 Early stopping for {model_name} — target {TARGET_ACC:.2f} reached at epoch {epoch}")
            break

        epoch += 1

    if epoch > EPOCH_LIMIT:
        print(f"⚠️ Max epoch limit ({EPOCH_LIMIT}) reached for {model_name} without hitting target accuracy.")

    # ---- Final Evaluation and Saving ----
    duration = time.time() - start_time

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    cm_path = f"results/{model_name}_transfer_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Saved confusion matrix to {cm_path}")

    retrained_path = f"models/{model_name}_retrained.pth"
    torch.save(model.state_dict(), retrained_path)
    print(f"💾 Saved retrained model to {retrained_path}")

    eval_results.append({
        "model": model_name,
        "accuracy": best_acc,
        "transfer_epochs": epoch,
        "eval_time_sec": duration
    })

# ---- Save Summary ----
df = pd.DataFrame(eval_results)
print("\n📊 Transfer Learning Summary:")
print(df.to_string(index=False))
df.to_csv("transfer_results.csv", index=False)
