import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model import get_model
from train import train
import torchvision.transforms.functional as TF
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from sklearn.metrics import (
accuracy_score, precision_score, recall_score,
f1_score, confusion_matrix, roc_auc_score
)
import json
from collections import defaultdict
import time

import os
os.makedirs("results/trained_models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


# --- SETTINGS ---
models = [
    #"efficientnetb0", "efficientnetb1", "efficientnetb2", 
    #"efficientnetb3", "efficientnetb4", "efficientnetb5", 
    "efficientnetb6",

    #"efficientnetv2_s",
    "efficientnetv2_m",
    #"efficientnetv2_l",
    
    #"resnet18",
    #"resnet34",
    #"resnet50",
    #"resnet101",
    #"resnext50_32x4d",
    #"resnext101_32x8d",

    #"densenet121",
    #"densenet169",
    #"densenetse",

    #"vit_b_16",
    #"deit_base_patch16_224",
    #"beit_base_patch16_224",
    
    #"swin_t",
    #"swin_s",
    #"swin_b",

    #"convnext_tiny",
    #"convnext_small",
    #"convnext_base",

    #"mobilenet_v3_large",
    #"efficientnet_lite0"
]

num_classes = 5
epochs = 15
batch_size = 16
resize = 224
pat = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

# --- DATA ---
input_size_dict = {
    "efficientnetb0": 224,
    "efficientnetb1": 240,
    "efficientnetb2": 288,
    "efficientnetb3": 300,
    "efficientnetb4": 380,
    "efficientnetb5": 456,
    "efficientnetb6": 528,
    "efficientnetb7": 600,
    "efficientnetv2_s": 384,
    "efficientnetv2_m": 480,
    "efficientnetv2_l": 480,
    "default": 224
}


all_metrics = []

for model_name in models:

    print(f"========> Training model: {model_name} ================================================================================")


    # Adjust batch size per model
    heavy_models = [
        "efficientnetb5", "efficientnetb6", "efficientnetb7",
        "efficientnetv2_m", "efficientnetv2_l", "convnext_base", "swin_b", "resnext101_32x8d"
    ]
    if model_name in heavy_models:
        batch_size = 2
    elif model_name == "efficientnetv2_m":
        batch_size = 4
    else:
        batch_size = 8

    # --- Resize and Transform ---
    resize = input_size_dict.get(model_name, input_size_dict["default"])

    try:
        start_time = time.time()
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor()
        ])

        # --- DATASETS and LOADERS ---
        train_set = ImageFolder("train_cnn/data/processed/train", transform=transform)
        val_set = ImageFolder("train_cnn/data/processed/val", transform=transform)
        test_set = ImageFolder("train_cnn/data/processed/test", transform=transform)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
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
        model_save_path = f"results/trained_models/model_{model_name}_{timestamp}.pth"

        # Train and save
        history, cm = train(model, train_loader, val_loader, device, epochs=epochs,
                            class_weights=weights_tensor, patience = pat, save_path=model_save_path)

        print(f"Training complete. Model saved to: {model_save_path}")

        # --- PLOT ---
        plt.figure(figsize=(8, 5))
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"results/plots/loss_curve_{model_name}.png")
        plt.close()

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_set.classes, yticklabels=train_set.classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"results/plots/confusion_matrix_{model_name}.png")
        plt.close()

        print("Plots saved in 'results/plots'.")

        # --- EVALUATE ---

        # Collect all predictions and ground truths
        y_true, y_pred, y_prob = [], [], []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                y_true.append(labels.item())
                y_pred.append(preds.item())
                y_prob.append(probs.cpu().numpy()[0])

        # Convert for metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # Multiclass metrics
        metrics = defaultdict(dict)
        metrics["model"] = model_name
        metrics["timestamp"] = timestamp
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["per_class_precision"] = dict(zip(test_set.classes, precision_score(y_true, y_pred, average=None, zero_division=0).tolist()))
        metrics["per_class_recall"] = dict(zip(test_set.classes, recall_score(y_true, y_pred, average=None, zero_division=0).tolist()))
        metrics["per_class_f1"] = dict(zip(test_set.classes, f1_score(y_true, y_pred, average=None, zero_division=0).tolist()))

        # Specificity per class
        cm = confusion_matrix(y_true, y_pred)
        specificity = []
        for i in range(num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity.append(tn / (tn + fp + 1e-6))  # avoid divide by zero

        metrics["specificity_per_class"] = dict(zip(test_set.classes, specificity))

        # AUC per class (requires one-hot labels)
        try:
            y_true_onehot = np.zeros_like(y_prob)
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            aucs = roc_auc_score(y_true_onehot, y_prob, average=None, multi_class="ovr")
            metrics["auc_per_class"] = dict(zip(test_set.classes, aucs.tolist()))
        except ValueError:
            metrics["auc_per_class"] = "not_computable"

        # Save metrics as JSON
        metrics_path = f"results/metrics_{model_name}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        #print(f"Saved metrics to {metrics_path}")
        end_time = time.time()
        total_time = end_time - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f" => {model_name} runtime: {hours}h {minutes}m {seconds}s\n")

    except Exception as e:
        print(f"Error with {model_name} : {e}")
        continue  

with open("results/bALL.json", "w") as f:
    json.dump(all_metrics, f, indent=4)


