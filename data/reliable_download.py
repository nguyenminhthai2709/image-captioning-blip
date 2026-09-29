"""
Resilient Downloader with automatic retries and resume for Flickr8k Dataset.
"""

import sys
import os
import time
import zipfile
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import setup_logger

logger = setup_logger("ResilientDownloader")

DATASET_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
TARGET_FILE = ROOT / "data" / "flickr8k" / "Flickr8k_Dataset.zip"
EXTRACT_DIR = ROOT / "data" / "flickr8k"


def download_with_resume(url: str, output_path: Path, max_retries: int = 50) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get total remote size
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    logger.info(f"Target file: {output_path.name} | Total size: {total_size / (1024*1024):.2f} MB")

    retries = 0
    pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading Flickr8k Images")

    while retries < max_retries:
        downloaded = output_path.stat().st_size if output_path.exists() else 0
        pbar.n = downloaded
        pbar.refresh()

        if downloaded >= total_size:
            logger.info("Download completed 100%!")
            break

        headers = {
            "Range": f"bytes={downloaded}-{total_size - 1}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as response, open(output_path, "ab") as f:
                while True:
                    chunk = response.read(512 * 1024)  # 512 KB
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        except Exception as e:
            retries += 1
            logger.warning(f"Connection interrupted: {e}. Retrying ({retries}/{max_retries}) in 2s ...")
            time.sleep(2)

    pbar.close()


def extract_and_organize(zip_path: Path, target_dir: Path) -> None:
    logger.info(f"Extracting {zip_path.name} to {target_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    logger.info("Extraction complete.")

    images_dir = target_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Move all jpgs from subfolder to Images/
    for sub in [target_dir / "Flicker8k_Dataset", target_dir / "Flickr8k_Dataset"]:
        if sub.exists():
            for f in sub.glob("*.jpg"):
                dest = images_dir / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
            shutil.rmtree(sub, ignore_errors=True)

    # Organize splits
    splits_dir = target_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        ("Flickr_8k.trainImages.txt", "train_images.txt"),
        ("Flickr_8k.devImages.txt", "val_images.txt"),
        ("Flickr_8k.testImages.txt", "test_images.txt")
    ]:
        if (target_dir / src).exists():
            shutil.copy(str(target_dir / src), str(splits_dir / src))
            shutil.copy(str(target_dir / src), str(splits_dir / dst))

    # Generate standard captions.txt from token file
    token_file = target_dir / "Flickr8k.token.txt"
    if token_file.exists():
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
                    img = img_raw.split("#")[0].strip()
                    clean_cap = cap.replace('"', '""')
                    rows.append(f'"{img}","{clean_cap}"')
        with open(captions_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")

    # Count real files
    all_imgs = list(images_dir.glob("*.jpg"))
    logger.info(f"SUCCESS: {len(all_imgs)} real JPEG images extracted into {images_dir}")

    # Remove zip file
    if zip_path.exists():
        zip_path.unlink()
        logger.info("Cleaned up zip archive.")


if __name__ == "__main__":
    download_with_resume(DATASET_URL, TARGET_FILE)
    extract_and_organize(TARGET_FILE, EXTRACT_DIR)
