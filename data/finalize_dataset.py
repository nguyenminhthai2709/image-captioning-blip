"""
Finalize Flickr8k dataset organization and verification.
"""

import shutil
from pathlib import Path

target_dir = Path("data/flickr8k")
splits_dir = target_dir / "splits"
splits_dir.mkdir(parents=True, exist_ok=True)

# 1. Copy standard split files
split_files = [
    ("Flickr_8k.trainImages.txt", "train_images.txt"),
    ("Flickr_8k.devImages.txt", "val_images.txt"),
    ("Flickr_8k.testImages.txt", "test_images.txt")
]

for src, dst in split_files:
    f = target_dir / src
    if f.exists():
        shutil.copy(str(f), str(splits_dir / src))
        shutil.copy(str(f), str(splits_dir / dst))

# 2. Format 40,460 captions
token_file = target_dir / "Flickr8k.token.txt"
captions_csv = target_dir / "captions.txt"
rows = ["image,caption"]

with open(token_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1) if "\t" in line else line.split(" ", 1)
        if len(parts) == 2:
            img_raw, cap = parts[0].strip(), parts[1].strip()
            img_name = img_raw.split("#")[0].strip()
            clean_cap = cap.replace('"', '""')
            rows.append(f'"{img_name}","{clean_cap}"')

with open(captions_csv, "w", encoding="utf-8") as f:
    f.write("\n".join(rows) + "\n")

# 3. Print verification
images_dir = target_dir / "Images"
all_images = list(images_dir.glob("*.jpg"))
train_imgs = (splits_dir / "train_images.txt").read_text(encoding="utf-8").strip().splitlines()
val_imgs = (splits_dir / "val_images.txt").read_text(encoding="utf-8").strip().splitlines()
test_imgs = (splits_dir / "test_images.txt").read_text(encoding="utf-8").strip().splitlines()

print("\n" + "=" * 65)
print("      FLICKR8K DATASET VERIFICATION & INSPECTION REPORT")
print("=" * 65)
print(f"[*] Total Real Images in Images/:     {len(all_images):,} images")
print(f"[*] Total Real Captions:              {len(rows)-1:,} captions")
print(f"[*] Average Captions per Image:       {(len(rows)-1)/len(all_images):.2f}")
print(f"[*] Standard Karpathy / Flickr8k Splits:")
print(f"    - Train Set:                      {len(train_imgs):,} images ({len(train_imgs)*5:,} captions)")
print(f"    - Val Set (Dev):                  {len(val_imgs):,} images ({len(val_imgs)*5:,} captions)")
print(f"    - Test Set:                       {len(test_imgs):,} images ({len(test_imgs)*5:,} captions)")
print("=" * 65 + "\n")
