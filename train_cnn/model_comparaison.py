"""
File: model_comparaison.py
Author: Youssef IBNOUALI
Date: August 2025

Description:
------------
This script analyzes and compares the performance of multiple trained CNN models for endoscopic image classification. It reads model evaluation metrics from
`results/tested_models/metrics/all_metrics.json` and generates:
1. A grouped bar plot showing global metrics (accuracy, precision, recall, F1)
2. A multi-class bar and line plot for per-class F1 scores
3. A CSV file summarizing all results
4. A console summary highlighting the best model by macro F1 score

Main Components:
----------------
- Load evaluation data from JSON
- Plot grouped bar chart of global metrics
- Plot combined per-class F1 scores
- Export results as a CSV summary
- Identify the top-performing model

Outputs:
--------
- `results/model_compar/model_global_metrics_barplot.png`  
- `results/model_compar/model_per_class_f1_barplot.png`  
- `results/model_compar/summary_metrics.csv`  
- Console printout of the best model by macro F1

Dependencies:
-------------
- matplotlib
- pandas
- numpy
- json
- os
- Python ≥ 3.6

Notes:
------
- All models must follow the same evaluation structure as expected in the JSON file.
- Colors are consistent per class across plots for easier comparison.
"""


import json
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# === Load data ===
with open("results/tested_models/metrics/all_metrics.json") as f:
    data = json.load(f)

models = [entry["model"] for entry in data]

# === Global metrics to compare ===
global_metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
global_values = {metric: [entry.get(metric, 0) for entry in data] for metric in global_metrics}

# === Plot 1: Global Metric Barplot ===
x = range(len(models))
width = 0.15

plt.figure(figsize=(11, 6))

colors = {
    "accuracy": "purple",
    "macro_precision": "orange",
    "macro_recall": "green",
    "macro_f1": "red"
}

for i, metric in enumerate(global_metrics):
    offset = (i - 1.5) * width
    values = global_values[metric]
    plt.bar([xi + offset for xi in x], values, width=width, label=metric, color=colors[metric])
    
    # Max point
    max_idx = np.argmax(values)
    max_value = values[max_idx]
    plt.plot(x[max_idx] + offset, max_value, marker="*", markersize=15, color="black", label=f"max metric" if i == 0 else "")


plt.xticks(x, models, rotation=15)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Global Metrics per Model")
plt.legend()
plt.tight_layout()
os.makedirs("results/model_compar", exist_ok=True)
plt.savefig("results/model_compar/model_global_metrics_barplot.png")
plt.show()


# === Plot 2: Per-Class F1 Scores ===
per_class = list(data[0]["per_class_f1"].keys())
per_class_values = {
    cls: [entry["per_class_f1"].get(cls, 0) for entry in data]
    for cls in per_class
}

class_colors = {
    "Normal": "#2ca02c", 
    "AG": "#1f77b4",    
    "IM": "#ffc013",   
    "Dysplasia": "#d62728",
    "Cancer": "#9467bd", 
}

plt.figure(figsize=(12, 6))
x = np.arange(len(models))
width = 0.12

# Barplot with offset and colors for each class
for i, cls in enumerate(per_class):
    offset = (i - (len(per_class) - 1) / 2) * width
    plt.bar(
        [xi + offset for xi in x],
        per_class_values[cls],
        width=width,
        alpha=0.5,
        color=class_colors[cls]
    )

# Line plot on top with same colors
for cls in per_class:
    values = per_class_values[cls]
    plt.plot(
        x,
        values,
        marker="o",
        linestyle="-",
        label=cls,
        color=class_colors[cls]
    )

plt.xticks(x, models, rotation=15, ha="right")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)
plt.title("Per-Class F1 Score per Model")
plt.legend(title="Classes")
plt.tight_layout()
plt.savefig("results/model_compar/model_per_class_f1_barplot.png")
plt.show()


# === CSV Export ===
df_rows = []
for entry in data:
    row = {
        "model": entry["model"],
        "accuracy": f"{entry['accuracy'] * 100:.2f}%",
        "macro_precision": f"{entry['macro_precision'] * 100:.2f}%",
        "macro_recall": f"{entry['macro_recall'] * 100:.2f}%",
        "macro_f1": f"{entry['macro_f1'] * 100:.2f}%"
    }
    for cls in per_class:
        f1 = entry["per_class_f1"].get(cls, 0)
        row[f"f1_{cls}"] = f"{f1 * 100:.2f}%"
    df_rows.append(row)

df = pd.DataFrame(df_rows)
os.makedirs("results/model_compar", exist_ok=True)
df.to_csv("results/model_compar/summary_metrics.csv", index=False)
print("CSV saved to results/model_compar/summary_metrics.csv")

# === Highlight Top Model ===
best_model = max(data, key=lambda m: m["macro_f1"])
print(f"\nBest model based on macro F1: {best_model['model']}")
print(f"  Accuracy      : {best_model['accuracy']:.4f}")
print(f"  Macro F1 Score: {best_model['macro_f1']:.4f}")
