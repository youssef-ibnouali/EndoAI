"""
File: extract_patchs.py
Author: Youssef IBNOUALI
Date: August 2025

Description:
------------
This script implements a semi-automatic patch extraction algorithm for endoscopic images.
It allows a user to select an image via a file dialog, then:
- Analyzes the image using entropy, intensity, and sharpness filters
- Selects high-information patches using a sliding window approach with NMS
- Classifies each patch using a pretrained CNN (e.g., EfficientNetB4)
- Saves the extracted patches into the `train_cnn/data/raw/` folder
- Visualizes patch locations and predicted labels in an annotated image

Use Case:
---------
After patch extraction, the user can manually sort patches into diagnosis folders
(AG, IM, Normal, etc.) for further training or supervised analysis.

Main Components:
----------------
- compute_entropy(): Local entropy filtering
- nms(): Non-maximum suppression on patch candidates
- extract_from_selected_image(): Main function to load, process, predict, and export patches

Output:
-------
- Patch files saved to: train_cnn/data/raw/
- Annotated overlay image shown and saved to: results/annotated_<image>.png

Dependencies:
-------------
- OpenCV
- PyTorch
- torchvision
- matplotlib
- scikit-image
- Tkinter (for file selection)
"""


import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from tkinter import Tk, filedialog
from skimage.filters.rank import entropy
from skimage.morphology import square
from skimage.util import img_as_ubyte
from model import get_model
from matplotlib import pyplot as plt

# === Utilitaires ===
def compute_entropy(gray_img):
    gray_uint8 = img_as_ubyte(gray_img)
    return entropy(gray_uint8, square(9))

def iou(a, b):
    xa1, ya1, xa2, ya2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    xb1, yb1, xb2, yb2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter_area = inter_w * inter_h
    union_area = a[2]*a[3] + b[2]*b[3] - inter_area
    return inter_area / union_area if union_area > 0 else 0

def nms(coords_scores, patch_size, iou_thresh=0.03):
    coords_scores.sort(key=lambda x: -x[2])
    final = []
    for c in coords_scores:
        x, y, score = c
        rect = [x, y, patch_size, patch_size]
        if all(iou(rect, r) <= iou_thresh for r in final):
            final.append(rect)
    return final

def tenengrad(patch_gray):
    gx = cv2.Sobel(patch_gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(patch_gray, cv2.CV_64F, 0, 1)
    grad_magnitude = np.sqrt(gx**2 + gy**2)
    return np.mean(grad_magnitude)
def is_blurry(patch_gray, tenengrad_thresh=20, lap_var_thresh=150):
    sharpness = tenengrad(patch_gray)
    lap_var = cv2.Laplacian(patch_gray, cv2.CV_64F).var()
    return sharpness < tenengrad_thresh or lap_var < lap_var_thresh

def extract_from_selected_image(model_name="efficientnetb4"):
    patch_size = 200
    step = 5
    class_names = ['AG', 'Cancer', 'Dysplasia', 'IM', 'Normal']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Choose image ===
    root = Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(title="Choose an image", filetypes=[("Image files", "*.png;*.jpg")])
    if not image_path:
        print("No image selected !")
        return

    basename = os.path.splitext(os.path.basename(image_path))[0]
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(8,8)).apply(gray)

    low_thresh = np.percentile(gray_eq, 18)
    high_thresh = np.percentile(gray_eq, 99.6)

    entropy_img = compute_entropy(gray_eq)
    entropy_norm = cv2.normalize(entropy_img, None, 0, 1, cv2.NORM_MINMAX)
    entropy_bin = entropy_norm > (np.mean(entropy_norm) * 0.5)
    intensity_mask = (gray > low_thresh) & (gray < high_thresh)
    combined_mask = np.logical_and(entropy_bin, intensity_mask)
    combined_mask = cv2.dilate(combined_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)

    a = np.percentile(gray_eq, 99.95)
    b = np.percentile(gray_eq, 6.5)

    # === Patch sélection ===
    candidates = []
    for y in range(0, gray.shape[0] - patch_size + 1, step):
        for x in range(0, gray.shape[1] - patch_size + 1, step):
            region = combined_mask[y:y+patch_size, x:x+patch_size]
            patch_gray = gray[y:y+patch_size, x:x+patch_size]
            if patch_gray.max() > a or patch_gray.min() < b:
                continue
            if region.mean() < 0.8:
                continue
            #if is_blurry(patch_gray): continue
            score = region.mean() * entropy_img[y:y+patch_size, x:x+patch_size].mean()
            candidates.append((x, y, score))

    final_coords = nms(candidates, patch_size)

    # === Modèle
    model = get_model(model_name, num_classes=5)
    model_path = "results/model_20250703_1158.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    batch = []
    for x, y, _, _ in final_coords:
        patch = img_rgb[y:y+patch_size, x:x+patch_size]
        pil = transforms.ToPILImage()(patch)
        tensor = transform(pil)
        batch.append(tensor)

    inputs = torch.stack(batch).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu().numpy()

    # === Sauver les patchs et dessiner
    img_draw = img_rgb.copy()

    for i, ((x, y, _, _), pred) in enumerate(zip(final_coords, preds), 1):
        label = class_names[pred]
        
        # === Patch brut, extrait avant dessin ===
        patch = img_rgb[y:y+patch_size, x:x+patch_size] 
        
        # === Sauvegarder le patch propre ===
        patch_name = f"{basename}_ipatch_{i}.png"
        cv2.imwrite(os.path.join("train_cnn/data/raw", patch_name), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

        # === Ensuite dessiner sur img_draw ===
        color = {
            'AG': (0, 255, 255), 'Cancer': (255, 0, 255),
            'Dysplasia': (255, 0, 0), 'IM': (255, 255, 0),
            'Normal': (0, 255, 0)
        }[label]
        cv2.rectangle(img_draw, (x, y), (x+patch_size, y+patch_size), color, 2)
        cv2.putText(img_draw, f"{i} · {label}", (x+5, y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


    print(f"✅ {len(preds)} patches saved to data/raw/")
    out_path = os.path.join("results", f"annotated_{basename}.png")
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_draw)
    plt.axis("off")
    plt.title(f"{basename} - {len(preds)} patches")
    plt.show()



extract_from_selected_image("efficientnetb4")