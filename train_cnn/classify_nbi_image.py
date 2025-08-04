import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.filters.rank import entropy
from skimage.morphology import square
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from train_cnn.model import get_model
import torch.nn.functional as F
import os

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

def nms(coords_scores, patch_size, iou_thresh=0.01):
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


def classify_nbi_image(image_path, model_name="efficientnetb4"):
    patch_size = 200
    step = 5
    iou_thresh = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['AG', 'Cancer', 'Dysplasia', 'IM', 'Normal']

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    
    low_thresh = np.percentile(gray_eq, 18)
    high_thresh = np.percentile(gray_eq, 99.6)

    # Entropy filtering
    entropy_img = compute_entropy(gray_eq)
    entropy_norm = cv2.normalize(entropy_img, None, 0, 1, cv2.NORM_MINMAX)
    entropy_bin = entropy_norm > (np.mean(entropy_norm)* 0.5)
    intensity_mask = (gray > low_thresh) & (gray < high_thresh)
    combined_mask = np.logical_and(entropy_bin, intensity_mask)
    combined_mask = cv2.dilate(combined_mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1)

    #cv2.imwrite("results/debug_mask.png", (combined_mask * 255).astype(np.uint8))
    #cv2.imwrite("results/debug_entropy.png", (entropy_norm * 255).astype(np.uint8))

    # Sliding window patch selection
    candidates = []
    a = np.percentile(gray_eq, 99.95)
    b = np.percentile(gray_eq, 6.75)
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            region = combined_mask[y:y+patch_size, x:x+patch_size]
            patch_gray = gray[y:y+patch_size, x:x+patch_size]
            if patch_gray.max() > a  or patch_gray.min() < b :
                continue
            valid_ratio = region.mean()
            if valid_ratio < 0.8:
                continue
            if is_blurry(patch_gray): continue
            score = valid_ratio * entropy_img[y:y+patch_size, x:x+patch_size].mean()
            candidates.append((x, y, score))

    #print(len(candidates))
    # Non-max suppression
    if not candidates:
        return None, {cls: 0 for cls in class_names}
    
    final_coords = nms(candidates, patch_size, iou_thresh)

    # Load model
    model_path = "results/model_20250730_1539.pth"
    model = get_model(model_name, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # Predict all patches
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
        probs = F.softmax(outputs, dim=1)
        confidences, pred_indices = torch.max(probs, dim=1)
        preds = pred_indices.cpu().numpy()
        confs = confidences.cpu().numpy()

    # Count class occurrences
    counts = dict.fromkeys(class_names, 0)
    for p in preds:
        counts[class_names[p]] += 1

    total = len(preds)
    scores = {cls: (counts[cls] / total) * 100 for cls in class_names}

    # Draw results
    img_draw = img_rgb.copy()
    for (x, y, _, _), p, conf in zip(final_coords, preds, confs):
        label = f"{class_names[p]} {conf*100:.1f}%"
        color = {
            'AG': (0, 255, 255), 'Cancer': (255, 0, 255),
            'Dysplasia': (255, 0, 0), 'IM': (255, 255, 0),
            'Normal': (0, 255, 0)
        }[class_names[p]]
        cv2.rectangle(img_draw, (x, y), (x+patch_size, y+patch_size), color, 2)
        cv2.putText(img_draw, label, (x+5, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


    os.makedirs("results", exist_ok=True)
    result_path = "results/result.png"
    cv2.imwrite(result_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))

    avg_confidence = float(np.mean(confs)) * 100 

    return result_path, scores, avg_confidence

