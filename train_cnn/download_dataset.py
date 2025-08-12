import os
import zipfile

# === SETTINGS ===
ZIP_DIR = os.path.join("train_cnn", "data_zip")
ZIP_NAME = "raw.zip"
TARGET_DIR = os.path.join("train_cnn", "data", "raw")

# === CREATE TARGET FOLDER IF NEEDED ===
os.makedirs(TARGET_DIR, exist_ok=True)

# === PATHS ===
zip_path = os.path.join(ZIP_DIR, ZIP_NAME)

# === CHECK IF ZIP EXISTS ===
if not os.path.exists(zip_path):
    raise FileNotFoundError(f"❌ ZIP file not found: {zip_path}")

# === UNZIP INTO train_cnn/data/raw/ ===
print(f"📂 Extracting {zip_path} into {TARGET_DIR} ...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(TARGET_DIR)

print(f"✅ Dataset ready in: {TARGET_DIR}")
