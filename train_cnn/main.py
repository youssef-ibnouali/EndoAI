"""
File: main.py
Author: Youssef IBNOUALI
Date: August 2025

Description:
------------
Main training and evaluation script for gastric endoscopy classification using deep learning.
This script loads data, trains a selected CNN model, evaluates it on the test set, visualizes results,
and exports performance metrics in both plots and JSON formats.

Key Functions and Features:
---------------------------
- Loads and preprocesses the dataset (train, val, test) using torchvision `ImageFolder`.
- Initializes a CNN architecture using `get_model()` from `model.py`.
- Trains the model using the `train()` function (with early stopping and class weighting).
- Saves training loss curves and confusion matrix.
- Computes detailed evaluation metrics (accuracy, precision, recall, F1-score, specificity).
- Saves a prediction preview on test images (`plot_test_predictions`).
- Exports metrics to `results/metrics_<model>_<timestamp>.json`.

Settings:
---------
- Model name (`model_name`)
- Input image resize dimensions
- Training hyperparameters (epochs, batch size, learning rate, patience)

Outputs:
--------
- Trained model checkpoint (in `results/`)
- Training/validation loss plot + confusion matrix (in `results/plots`)
- Test predictions visualization (in `results/`)
- Evaluation metrics JSON file (in `results/`)

Usage:
------
Run this script directly to train and evaluate a single model:
>>> python main.py

Make sure the training/validation/test datasets exist under:
>>> train_cnn/data/processed/{train,val,test}/

Dependencies:
-------------
- torch, torchvision
- sklearn
- matplotlib, seaborn, numpy
- model.py (defines all CNN variants)
- train.py (defines the training loop)
"""


import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import get_model
from train import train
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import json
import time
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    roc_auc_score, confusion_matrix
)  


# --- SETTINGS ---
model_name = "efficientnetb4"  
num_classes = 5
epochs = 100
batch_size = 16
resize = 380
patien = 10
l_r = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

start_time = time.time()

# --- DATA ---
transform = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor()
])

train_set = ImageFolder("train_cnn/data/processed/train", transform=transform)
val_set = ImageFolder("train_cnn/data/processed/val", transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_set = ImageFolder("train_cnn/data/processed/test", transform=transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

# --- MODEL ---
model = get_model(model_name, num_classes=num_classes)


labels = train_set.targets 
classes = np.unique(labels)

# Compute weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)
# Convert to tensor
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)


# --- TRAIN ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_save_path = f"results/model_{timestamp}.pth"

# Train and save
history, cm = train(model, train_loader, val_loader, device, epochs=epochs, lr=l_r,
                    class_weights=weights_tensor, patience=patien, save_path=model_save_path)

print(f"Training complete. Model saved to: {model_save_path}")


# --- PLOT ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  

# --- Courbe de perte ---
axes[0].plot(history["train_loss"], label="Train Loss")
axes[0].plot(history["val_loss"], label="Val Loss")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

# --- Matrice de confusion ---
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=train_set.classes, yticklabels=train_set.classes,
            ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

plt.tight_layout()
plt.savefig("results/plots/loss_and_confusion_matrix.png")
plt.show()
plt.close()

print("Plots saved in 'results/'.")


def plot_test_predictions(model, test_loader, class_names, device, save_path="results/sample_predictions.png"):
    model.eval()
    shown_classes = set()
    images_shown = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            true_label = labels.item()
            pred_label = preds.item()

            if true_label not in shown_classes:
                shown_classes.add(true_label)
                images_shown.append((images[0].cpu(), true_label, pred_label))

            if len(shown_classes) == len(class_names):
                break

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Plot
    fig, axes = plt.subplots(1, len(images_shown), figsize=(15, 5))
    for i, (img, true_label, pred_label) in enumerate(images_shown):
        img = TF.to_pil_image(img)
        axes[i].imshow(img)
        title = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
        if true_label == pred_label:
            axes[i].set_title(title, color="green")
        else:
            axes[i].set_title(title, color="red")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()


# --- TEST ACCURACY ---
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.2%}")


plot_test_predictions(model, test_loader, test_set.classes, device)
print("Test sample predictions saved.")

# --- METRICS JSON GENERATION ---
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
class_names = test_set.classes
n_classes = len(class_names)

# --- GLOBAL METRICS ---
accuracy = accuracy_score(all_labels, all_preds)
macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

# --- PER-CLASS METRICS ---
per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

# --- SPECIFICITY ---
cm = confusion_matrix(all_labels, all_preds)
specificity_per_class = []
for i in range(n_classes):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    specificity_per_class.append(specificity)

# --- SAVE METRICS TO JSON ---
results_dict = {
    "model": model_name,
    "timestamp": timestamp,
    "accuracy": accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "per_class_precision": dict(zip(class_names, per_class_precision.tolist())),
    "per_class_recall": dict(zip(class_names, per_class_recall.tolist())),
    "per_class_f1": dict(zip(class_names, per_class_f1.tolist())),
    "specificity_per_class": dict(zip(class_names, specificity_per_class))
}

json_path = f"results/metrics_{model_name}_{timestamp}.json"
with open(json_path, "w") as f:
    json.dump(results_dict, f, indent=4)

print(f"Evaluation metrics saved to: {json_path}")


end_time = time.time()
total_time = end_time - start_time

hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print(f"\n => Total runtime: {hours}h {minutes}m {seconds}s")