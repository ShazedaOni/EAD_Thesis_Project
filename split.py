# ================================
# 📂 Image Dataset Splitter
# ================================
# This script will:
# 1️⃣ Read images from "data/autistic" and "data/normal"
# 2️⃣ Split them into train/val/test sets (default: 70/15/15)
# 3️⃣ Create "dataset/train", "dataset/val", "dataset/test" folders
# 4️⃣ Copy images into their respective class subfolders

# ✅ Import Required Libraries
import os
import shutil
from pathlib import Path
import random
from math import floor

BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR / "Thesis testing data"   # ✅ correct folder name
OUTPUT_DIR = BASE_DIR / "dataset_split"

# Create output folders
splits = ['train', 'val', 'test']
classes = ['Autistic child', 'non-Autistic child']

for split in splits:
    for cls in classes:
        folder = OUTPUT_DIR / split / cls
        folder.mkdir(parents=True, exist_ok=True)

# ==========================
# STEP 2: SPLIT RATIOS
# ==========================
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# ==========================
# STEP 3: COPY FILES
# ==========================
for cls in classes:
    class_dir = SOURCE_DIR / cls
    images = [img for img in class_dir.iterdir() if img.is_file()]
    random.shuffle(images)

    total = len(images)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(img, OUTPUT_DIR / "train" / cls / img.name)
    for img in val_imgs:
        shutil.copy(img, OUTPUT_DIR / "val" / cls / img.name)
    for img in test_imgs:
        shutil.copy(img, OUTPUT_DIR / "test" / cls / img.name)

print("✅ Data split completed successfully!")
print(f"Total images processed per class:")
for cls in classes:
    count = len(list((SOURCE_DIR / cls).glob('*')))
    print(f"{cls}: {count}")
