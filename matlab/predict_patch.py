# predict_patch.py
import torch
from torchvision import transforms
from model import get_model
from PIL import Image
import sys

# === CONFIG ===
MODEL_PATH = "results/model_20250702_1256.pth"  
NUM_CLASSES = 5
CLASS_NAMES = ['AG', 'Cancer', 'Dysplasia', 'IM', 'Normal']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model("efficientnetb4", num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Define transform (NO normalization) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Load image path from MATLAB ===
img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# === Predict ===
with torch.no_grad():
    output = model(img_tensor)
    pred = output.argmax(1).item()

# === Print class label for MATLAB ===
print(CLASS_NAMES[pred])
