"""
File: train.py
Author: Youssef IBNOUALI
Date: August 2025

Description:
------------
Contains the training loop for supervised learning of convolutional neural networks (CNNs)
on gastric endoscopic image classification. Includes support for early stopping,
adaptive learning rate scheduling, class imbalance handling, and performance tracking.

Main Function:
--------------
train(model, train_loader, val_loader, device, epochs=10, lr=1e-3, patience=20,
      save_path="results/model.pth", class_weights=None)

Parameters:
-----------
- model         : PyTorch model to be trained
- train_loader  : DataLoader for the training set
- val_loader    : DataLoader for the validation set
- device        : 'cuda' or 'cpu'
- epochs        : Maximum number of training epochs (default: 10)
- lr            : Initial learning rate (default: 1e-3)
- patience      : Early stopping patience based on validation loss (default: 20)
- save_path     : Path to save the best model (default: "results/model.pth")
- class_weights : Optional tensor of class weights for imbalanced data

Returns:
--------
- history       : Dictionary containing lists of train/validation loss and accuracy
- cm            : Final confusion matrix (numpy array) from validation set

Features:
---------
- CrossEntropyLoss with optional class weights and label smoothing (train only)
- ReduceLROnPlateau scheduler for dynamic learning rate adjustment
- Early stopping based on best validation loss
- Tracking and returning of confusion matrix
- Automatic directory creation and model saving

Usage:
------
>>> from train import train
>>> history, cm = train(model, train_loader, val_loader, device, epochs=15)

Dependencies:
-------------
- torch
- sklearn.metrics (for confusion matrix)
- numpy
- os
"""

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import os

def train(model, train_loader, val_loader, device, epochs=10, lr=1e-3, patience = 20, save_path="results/model.pth",  class_weights=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1) if class_weights is not None else nn.CrossEntropyLoss(label_smoothing=0.1)
    val_criterion = nn.CrossEntropyLoss()  # no weights for validation


    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    model.to(device)

    best_val_loss = float("inf")
    trigger_times = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = train_criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = val_criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / total
        val_acc = correct / total
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss - 1e-4: 
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"  EarlyStopping patience {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("  Early stopping triggered.")
                break

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model_state if best_model_state else model.state_dict(), save_path)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return history, cm
