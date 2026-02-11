import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna
from PIL import Image
from torchvision import transforms
import timm

from models_factory import load_resnet50

# ---- Configuration ----
AUGMENTED_DIR = "data/ideal_subset"
RESULTS_BASE_DIR = "adv_results"
IMG_SIZE = 224
NUM_CLASSES = 671
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Image Transform ----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---- Classifier Wrapper ----
class TargetClassifier:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, image_np):
        tensor = transform(Image.fromarray(image_np)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            return pred.item(), probs.cpu()

# ---- Agent ----
class AdversarialMaskAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state):
        return np.random.uniform(0, 1, size=self.action_dim)

# ---- Environment ----
class AdversarialMaskEnv:
    def __init__(self, image, classifier, max_steps=20):
        self.classifier = classifier
        self.max_steps = max_steps
        self.h, self.w = IMG_SIZE, IMG_SIZE
        self.original_image = image
        self.reset()

    def reset(self):
        self.current_image = self.original_image.copy()
        self.step_count = 0
        self.history = [self.current_image.copy()]
        return np.array([0.0] * 5, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        cx, cy = int(action[0] * self.w), int(action[1] * self.h)
        radius = int(action[2] * min(self.h, self.w) * 0.25)
        intensity = float(action[3])

        modified = self.current_image.copy()
        if radius > 1:
            yy, xx = np.ogrid[:self.h, :self.w]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            noise = np.random.uniform(-intensity, intensity, modified.shape).astype(np.float32)
            modified[mask] = np.clip(modified[mask] + noise[mask], 0.0, 1.0)

        self.history.append(modified.copy())
        pred, probs = self.classifier.predict(self.original_image)
        pred_new, probs_new = self.classifier.predict(modified)
        reward = float(probs[0][pred]) - float(probs_new[0][pred])
        self.current_image = modified
        done = self.step_count >= self.max_steps
        return np.array([0.0] * 5, dtype=np.float32), reward, done, {"observations": self.history}

# ---- Load Model ----
def load_model_by_name(name, num_classes, device):
    if name == "resnet_cnn":
        model = load_resnet50(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    path = f"models/{name}.pth"
    model.load_state_dict(torch.load(path, map_location=device))
    return model

# ---- Load Images ----
def load_images(image_dir, max_images=10):
    images = []
    files = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for i, fname in enumerate(files[:max_images]):
        path = os.path.join(image_dir, fname)
        image = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        images.append((f"image_{i}", np.array(image), "flat", fname))
    return images

import matplotlib.pyplot as plt

# Hybrid visualization function to display best frame in real-time during training
def display_best_frame(frame, episode, reward):
    plt.imshow(frame)
    plt.title(f"Episode {episode} - Reward: {reward:.3f}")
    plt.axis("off")
    plt.pause(0.01)
    plt.clf()

# ---- Adversarial Testing Function ----
def run_adversarial_testing(images, model_name,
                            max_patch_radius=0.1,
                            max_noise=0.05,
                            max_steps=150,
                            save_outputs=False,
                            trial_id="000"):
    print(f"\n🚀 Running: radius={max_patch_radius:.3f}, noise={max_noise:.3f}, steps={max_steps}")
    results_dir = os.path.join(RESULTS_BASE_DIR, model_name)
    os.makedirs(results_dir, exist_ok=True)

    cumulative_reward = 0

    for sample_id, image, _, filename in images:
        model = load_model_by_name(model_name, NUM_CLASSES, DEVICE)
        classifier = TargetClassifier(model, DEVICE)
        env = AdversarialMaskEnv(image=image, classifier=classifier, max_steps=max_steps)
        agent = AdversarialMaskAgent(state_dim=5, action_dim=4)

        best_reward = float("-inf")
        best_observations = []
        plateau_counter = 0
        max_plateau = 30
        ep = 1
        max_episodes = 100

        trial_dir = os.path.join(results_dir, sample_id, f"trial_{trial_id}")
        os.makedirs(trial_dir, exist_ok=True)

        while plateau_counter < max_plateau and ep <= max_episodes:
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                action[2] = min(action[2], max_patch_radius)
                action[3] = min(action[3], max_noise)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state

            if total_reward > best_reward:
                best_reward = total_reward
                best_observations = info["observations"]
                display_best_frame(best_observations[-1], ep, total_reward) 
                plateau_counter = 0
            else:
                improvement = (total_reward - best_reward) / max(abs(best_reward), 1e-5)
                plateau_counter += 1 if improvement >= -0.05 else 0

            ep += 1

        cumulative_reward += best_reward

        if save_outputs:
            for i, frame in enumerate(best_observations):
                plt.imsave(os.path.join(trial_dir, f"frame_{i:02d}.png"), frame)

    avg_reward = cumulative_reward / len(images) if images else 0
    print(f"✅ Average reward: {avg_reward:.4f}")
    return avg_reward

# ---- Optuna Optimization ----
def optimize_patch_parameters(images, model_name, n_trials=10):
    def objective(trial):
        max_patch_radius = trial.suggest_float("max_patch_radius", 0.05, 0.25)
        max_noise = trial.suggest_float("max_noise", 0.01, 0.2)
        max_steps = trial.suggest_int("max_steps", 50, 300)

        trial_id = f"{trial.number:03d}"
        avg_reward = run_adversarial_testing(
            images=images,
            model_name=model_name,
            max_patch_radius=max_patch_radius,
            max_noise=max_noise,
            max_steps=max_steps,
            save_outputs=True,
            trial_id=trial_id
        )

        if avg_reward < -1e4:
            trial.report(avg_reward, step=0)
            raise optuna.TrialPruned()

        return avg_reward

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("\n🏆 Best Parameters Found:")
    for k, v in study.best_params.items():
        print(f"   ➤ {k}: {v:.4f}" if isinstance(v, float) else f"   ➤ {k}: {v}")
    print(f"   ➤ Best Avg Reward: {study.best_value:.3f}")

# ---- Main ----
if __name__ == "__main__":
    images = load_images(AUGMENTED_DIR, max_images=10)
    optimize_patch_parameters(images, model_name="resnet_cnn")
