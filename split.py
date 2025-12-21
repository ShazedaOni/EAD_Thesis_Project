import os
import shutil
import random

# source folders
SOURCE_DIR = "./"
CLASSES = ["autistic", "non_autistic"]

# destination folders
DEST_DIRS = ["train", "val", "test"]

# split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

random.seed(42)

# create destination folders
for folder in DEST_DIRS:
    for cls in CLASSES:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

for cls in CLASSES:
    files = os.listdir(os.path.join(SOURCE_DIR, cls))
    random.shuffle(files)

    total = len(files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for f in train_files:
        shutil.copy(
            os.path.join(SOURCE_DIR, cls, f),
            os.path.join("train", cls, f)
        )

    for f in val_files:
        shutil.copy(
            os.path.join(SOURCE_DIR, cls, f),
            os.path.join("val", cls, f)
        )

    for f in test_files:
        shutil.copy(
            os.path.join(SOURCE_DIR, cls, f),
            os.path.join("test", cls, f)
        )

print("✅ Dataset split completed!")
