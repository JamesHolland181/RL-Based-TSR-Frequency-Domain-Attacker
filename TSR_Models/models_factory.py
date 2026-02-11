# models_factory.py
import torch
import torch.nn as nn
from torchvision import models
from cnn_3spt import TrafficSignCNNFull
from SAG_VIT import SAGViTLISAClassifier


def load_resnet50(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_vit(num_classes):
    model = models.vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


# def load_cnn_3spt(path, num_classes, device):
#     model = TrafficSignCNNFull(num_classes=num_classes, use_stn=True).to(device)

#     # Load only weights that match current model
#     state_dict = torch.load(path, map_location=device)
#     filtered_state_dict = {
#         k: v for k, v in state_dict.items()
#         if k in model.state_dict() and v.shape == model.state_dict()[k].shape
#     }
#     model.load_state_dict(filtered_state_dict, strict=False)
#     model.eval()
#     return model

def load_cnn_3spt(path, num_classes, device):
    print(f"🔍 Loading CNN_3SPT with num_classes={num_classes}, path={path}")
    model = TrafficSignCNNFull(num_classes=num_classes, use_stn=True).to(device)

    state_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()

    # Compare shapes
    mismatches = []
    for k in state_dict:
        if k in model_dict:
            if state_dict[k].shape != model_dict[k].shape:
                mismatches.append((k, state_dict[k].shape, model_dict[k].shape))
        else:
            print(f"⚠️ Key from checkpoint not in model: {k}")

    if mismatches:
        print("❌ Mismatched layers:")
        for name, ckpt_shape, model_shape in mismatches:
            print(f" - {name}: checkpoint {ckpt_shape}, model {model_shape}")
    else:
        print("✅ All shapes match.")

    # Load only matching weights
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model  

def load_sag_vit(path, num_classes, device):
    model = SAGViTLISAClassifier(num_classes=num_classes).to(device)

    # Load only matching weights
    state_dict = torch.load(path, map_location=device)
    filtered_dict = {k: v for k, v in state_dict.items()
                     if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(filtered_dict, strict=False)

    model.eval()
    return model


def load_cnn_3spt_arch(num_classes, device):
    return TrafficSignCNNFull(num_classes=num_classes, use_stn=True).to(device)

def load_sag_vit_arch(num_classes, device):
    return SAGViTLISAClassifier(num_classes=num_classes).to(device)
