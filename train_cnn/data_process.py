import os
import random
import shutil
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from pathlib import Path
import albumentations as A
import numpy as np
import cv2

# -------- CONFIG --------
INPUT_ROOT = Path("train_cnn/data/raw")
AUGMENTED_ROOT = Path("train_cnn/data/processed")
SPLIT_ROOT = Path("train_cnn/data/split")
SPLIT_RATIOS = {
    "train": 0.65,
    "val": 0.25,
    "test": 0.1
}

# -------- CLEAN FOLDERS --------
import stat

def handle_remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clean_dir(path):
    if path.exists():
        shutil.rmtree(path, onerror=handle_remove_readonly)
    path.mkdir(parents=True, exist_ok=True)

# -------- AUGMENTATION --------

def elastic_transform(pil_img):
    transform = A.ElasticTransform(alpha=50, sigma=5, p=1.0)
    img_np = np.array(pil_img)
    augmented = transform(image=img_np)
    img_aug_np = augmented["image"]

    h, w = img_aug_np.shape[:2]
    crop_margin = int(min(h, w) * 0.05)

    # Vérifie que la taille restante est suffisante
    if h - 2 * crop_margin <= 0 or w - 2 * crop_margin <= 0:
        cropped = img_aug_np  # Pas de crop
    else:
        cropped = img_aug_np[
            crop_margin : h - crop_margin,
            crop_margin : w - crop_margin
        ]

    cropped_resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cropped_resized)


def augment_and_save(input_root, output_root):

    # Base transforms: original, horizontal flip, vertical flip
    base_transforms = [
        ("orig", lambda x: x),
        ("hflip", F.hflip),
        ("vflip", F.vflip),
    ]

    # Rotation transforms
    rotation_transforms = [
        ("", lambda x: x),  # no rotation (keep base as-is)
        ("rot90", lambda x: F.rotate(x, 90)),
        ("rot180", lambda x: F.rotate(x, 180)),
        ("rot270", lambda x: F.rotate(x, 270)),
    ]

    normalize = transforms.Compose([
        transforms.ToTensor()
    ])

    for class_dir in input_root.iterdir():
        if class_dir.is_dir():
            out_class_dir = output_root / class_dir.name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in class_dir.glob("*"):
                try:
                    img = Image.open(img_path).convert("RGB")

                    for base_suffix, base_func in base_transforms:
                        base_img = base_func(img)

                        for rot_suffix, rot_func in rotation_transforms:
                            variant_img = rot_func(base_img)
                            suffix = f"{base_suffix}_{rot_suffix}".strip("_")

                            # Save variant
                            filename = f"{img_path.stem}_{suffix}.png"
                            save_path = out_class_dir / filename
                            F.to_pil_image(normalize(variant_img)).save(save_path)

                            # Apply Elastic on that variant
                            elastic_img = elastic_transform(variant_img)
                            elastic_filename = f"{img_path.stem}_{suffix}_elastic.png"
                            elastic_path = out_class_dir / elastic_filename
                            F.to_pil_image(normalize(elastic_img)).save(elastic_path)

                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")


# -------- SPLITTING --------
def split_dataset(processed_root, output_root, split_ratios):
    output_root.mkdir(parents=True, exist_ok=True)
    for split in split_ratios.keys():
        for class_name in os.listdir(processed_root):
            split_path = output_root / split / class_name
            split_path.mkdir(parents=True, exist_ok=True)

    for class_name in os.listdir(processed_root):
        class_dir = processed_root / class_name
        images = list(class_dir.glob("*.png"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(split_ratios["train"] * n_total)
        n_val = int(split_ratios["val"] * n_total)
        n_test = n_total - n_train - n_val

        split_counts = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        for split, files in split_counts.items():
            for f in files:
                shutil.copy(f, output_root / split / class_name / f.name)

# -------- MAIN --------
if __name__ == "__main__":

    print("Cleaning previous outputs...")
    clean_dir(SPLIT_ROOT)
    clean_dir(AUGMENTED_ROOT)

    print(f"Total raw images: {sum(1 for _ in INPUT_ROOT.rglob('*.png'))}")

    # Step 1: Split raw images
    print("Splitting dataset from raw...")
    split_dataset(INPUT_ROOT, SPLIT_ROOT, SPLIT_RATIOS)

    # Step 2: Augment only train
    print("Augmenting training set only...")
    augment_and_save(SPLIT_ROOT / "train", AUGMENTED_ROOT / "train")
    print(f"=>augmented from {sum(1 for _ in (Path("train_cnn/data/raw/train")).rglob('*.png'))} to {sum(1 for _ in (Path("train_cnn/data/processed/train")).rglob('*.png'))}")
    # Step 3: Copy val and test sets unchanged
    for split in ["val", "test"]:
        for class_dir in (SPLIT_ROOT / split).iterdir():
            dest = AUGMENTED_ROOT / split / class_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img_file in class_dir.glob("*"):
                shutil.copy(img_file, dest)

    print(f"Total final images: {sum(1 for _ in AUGMENTED_ROOT.rglob('*.png'))}")
    print("Dataset ready in 'train_cnn/data/processed'")