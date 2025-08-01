# predict_batch_folder.py
import os
import torch
from torchvision import transforms
from PIL import Image
from model import get_model

MODEL_PATH = "results/model_20250702_1256.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['AG', 'Cancer', 'Dysplasia', 'IM', 'Normal']
NUM_CLASSES = 5

# Load model once
model = get_model("efficientnetb4", num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load all images in temp_patches/
patch_dir = "temp_patches"
images = []
file_list = sorted(f for f in os.listdir(patch_dir) if f.endswith(".png"))

for filename in file_list:
    path = os.path.join(patch_dir, filename)
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img)
    images.append(img_tensor)

batch = torch.stack(images).to(DEVICE)

# Predict
with torch.no_grad():
    outputs = model(batch)
    preds = outputs.argmax(1).cpu().numpy()

# Write predictions
with open(os.path.join(patch_dir, "predictions.txt"), "w") as f:
    for p in preds:
        f.write(f"{CLASS_NAMES[p]}\n")
