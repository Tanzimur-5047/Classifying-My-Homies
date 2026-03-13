import os
import shutil
import random

def split_data(train_folder, val_folder, split=0.2):


    images = os.listdir(train_folder)


    random.shuffle(images)


    val_count = int(len(images) * split)


    val_images = images[:val_count]

    for img in val_images:

        src = os.path.join(train_folder, img)

        dst = os.path.join(val_folder, img)

        shutil.move(src, dst)

    print(f"Moved {val_count} images to {val_folder}")
    print(f"Remaining in {train_folder}: {len(images) - val_count}")


if __name__ == "__main__":  
    pairs = [
        ("data/train/Aninda",  "data/val/Aninda"),
        ("data/train/Himel",   "data/val/Himel"),
        ("data/train/Sukomal", "data/val/Sukomal"),
    ]

    for train_folder, val_folder in pairs:
        print(f"\n--- Splitting {train_folder} ---")
        split_data(train_folder, val_folder)

    print("\nDone! Train/val split complete.")
