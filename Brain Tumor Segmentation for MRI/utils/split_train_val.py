import os
import shutil
import random
from pathlib import Path

def split_dataset(images_dir, masks_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'masks'), exist_ok=True)

    filenames = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    random.shuffle(filenames)
    val_count = int(len(filenames) * val_ratio)
    val_set = set(filenames[:val_count])

    for fname in filenames:
        subset = 'val' if fname in val_set else 'train'
        shutil.copy2(os.path.join(images_dir, fname), os.path.join(output_dir, subset, 'images', fname))
        shutil.copy2(os.path.join(masks_dir, fname), os.path.join(output_dir, subset, 'masks', fname))

    print(f"Split complete: {len(filenames) - val_count} train / {val_count} val")

if __name__ == "__main__":
    split_dataset(
        images_dir='data/processed/images',
        masks_dir='data/processed/masks',
        output_dir='data/processed'
    )

