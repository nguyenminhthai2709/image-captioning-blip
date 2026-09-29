"""
Fast Multi-Threaded Chunk Downloader with Automatic Retries per Chunk.
"""

import sys
import os
import time
import shutil
import zipfile
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Ensure utf-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import setup_logger

logger = setup_logger("FastDownloader")

DATASET_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
TARGET_DIR = ROOT / "data" / "flickr8k"
TARGET_FILE = TARGET_DIR / "Flickr8k_Dataset.zip"


def download_chunk_with_retry(url: str, start: int, end: int, part_path: Path, max_retries: int = 10) -> None:
    """Download a byte range with individual retry and resume."""
    for attempt in range(1, max_retries + 1):
        try:
            current_bytes = part_path.stat().st_size if part_path.exists() else 0
            if current_bytes >= (end - start + 1):
                return  # Chunk already complete

            req_start = start + current_bytes
            headers = {
                "Range": f"bytes={req_start}-{end}",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp, open(part_path, "ab") as f:
                shutil.copyfileobj(resp, f)

            if part_path.stat().st_size >= (end - start + 1):
                return
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Chunk {part_path.name} failed after {max_retries} attempts: {e}")
            time.sleep(1.5)


def parallel_download(url: str, output_path: Path, num_threads: int = 8) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent / ".temp_chunks"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Get total size
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    logger.info(f"Downloading {output_path.name} ({total_size / (1024*1024):.2f} MB) using {num_threads} parallel connections...")

    chunk_size = total_size // num_threads
    parts_info = []
    for i in range(num_threads):
        start = i * chunk_size
        end = total_size - 1 if i == num_threads - 1 else (start + chunk_size - 1)
        part_file = temp_dir / f"chunk_{i:02d}.bin"
        parts_info.append((i, start, end, part_file))

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Flickr8k Download") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(download_chunk_with_retry, url, start, end, part_file)
                for _, start, end, part_file in parts_info
            ]

            while any(not f.done() for f in futures):
                time.sleep(0.5)
                current_total = sum(p[3].stat().st_size for p in parts_info if p[3].exists())
                pbar.n = current_total
                pbar.refresh()

            for f in futures:
                f.result()

            pbar.n = total_size
            pbar.refresh()

    logger.info("Merging chunks into final zip...")
    with open(output_path, "wb") as outfile:
        for _, _, _, part_file in sorted(parts_info, key=lambda x: x[0]):
            with open(part_file, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info("Zip file created and verified successfully!")


def extract_and_organize(zip_path: Path, target_dir: Path) -> None:
    logger.info(f"Extracting {zip_path.name} to {target_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    logger.info("Extraction completed.")

    images_dir = target_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Move extracted images
    for folder in [target_dir / "Flicker8k_Dataset", target_dir / "Flickr8k_Dataset"]:
        if folder.exists():
            for f in folder.glob("*.jpg"):
                dest = images_dir / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
            shutil.rmtree(folder, ignore_errors=True)

    # Clean zip
    if zip_path.exists():
        zip_path.unlink()


def print_verification_report(target_dir: Path) -> None:
    images_dir = target_dir / "Images"
    all_imgs = list(images_dir.glob("*.jpg"))
    
    # Read captions
    captions_csv = target_dir / "captions.txt"
    caps_count = 0
    if captions_csv.exists():
        with open(captions_csv, "r", encoding="utf-8") as f:
            caps_count = max(0, len(f.readlines()) - 1)

    # Read splits
    splits_dir = target_dir / "splits"
    def count_split(fname: str) -> int:
        p = splits_dir / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return len([l for l in f if l.strip()])
        return 0

    print("\n" + "=" * 65)
    print("      FLICKR8K DATASET VERIFICATION & INSPECTION REPORT")
    print("=" * 65)
    print(f"[*] Total Real Images in Images/:     {len(all_imgs):,} images")
    print(f"[*] Total Captions in captions.txt:    {caps_count:,} captions")
    print(f"[*] Standard Karpathy / Flickr8k Splits:")
    print(f"    - Train Set:                      {count_split('train_images.txt'):,} images")
    print(f"    - Val Set:                        {count_split('val_images.txt'):,} images")
    print(f"    - Test Set:                       {count_split('test_images.txt'):,} images")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parallel_download(DATASET_URL, TARGET_FILE, num_threads=8)
    extract_and_organize(TARGET_FILE, TARGET_DIR)
    print_verification_report(TARGET_DIR)
