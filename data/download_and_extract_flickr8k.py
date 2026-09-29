"""
High-Speed Multi-Threaded Downloader, Extractor and Validator for Flickr8k Dataset.

Sources:
- Text: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
- Dataset: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
"""

import sys
import os
import shutil
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Ensure utf-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.utils import setup_logger

logger = setup_logger("Flickr8kDownloader")

TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
DATASET_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"


def download_chunk(url: str, start: int, end: int, part_path: Path) -> None:
    """Download a byte range chunk."""
    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp, open(part_path, "wb") as f:
        shutil.copyfileobj(resp, f)


def download_file_multithreaded(url: str, output_path: Path, num_threads: int = 12) -> None:
    """
    Download a file using parallel HTTP Range connections for maximum bandwidth.
    """
    logger.info(f"Connecting to {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))

    if output_path.exists() and output_path.stat().st_size == total_size:
        logger.info(f"File already exists with full size ({total_size} bytes): {output_path} (skipping download)")
        return
    elif output_path.exists():
        output_path.unlink()

    if total_size <= 0 or num_threads <= 1:
        # Fallback to single thread
        with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        return

    logger.info(f"Downloading {output_path.name} ({total_size / (1024*1024):.2f} MB) using {num_threads} parallel threads...")
    
    chunk_size = total_size // num_threads
    parts_info = []
    temp_dir = output_path.parent / f".tmp_{output_path.name}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_threads):
        start = i * chunk_size
        end = total_size - 1 if i == num_threads - 1 else (start + chunk_size - 1)
        part_file = temp_dir / f"part_{i}.tmp"
        parts_info.append((i, start, end, part_file))

    with tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _, start, end, part_file in parts_info:
                f = executor.submit(download_chunk, url, start, end, part_file)
                futures.append(f)

            # Monitor progress
            import time
            while any(not f.done() for f in futures):
                time.sleep(0.5)
                current_downloaded = sum(p[3].stat().st_size for p in parts_info if p[3].exists())
                pbar.n = current_downloaded
                pbar.refresh()

            for f in futures:
                f.result()  # Raise exception if any worker failed

            pbar.n = total_size
            pbar.refresh()

    logger.info("Merging downloaded chunks into final archive...")
    with open(output_path, "wb") as outfile:
        for _, _, _, part_file in sorted(parts_info, key=lambda x: x[0]):
            with open(part_file, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info(f"Download completed successfully: {output_path.name}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip archive."""
    logger.info(f"Extracting {zip_path.name} to {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    logger.info(f"Extraction of {zip_path.name} finished.")


def organize_flickr8k(target_dir: Path) -> None:
    """
    Organize extracted files into standard layout:
    - data/flickr8k/Images/
    - data/flickr8k/captions.txt
    - data/flickr8k/splits/
    """
    logger.info("Organizing Flickr8k directory structure...")
    images_dir = target_dir / "Images"
    splits_dir = target_dir / "splits"
    images_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Move images from subfolders into Images/
    possible_img_dirs = [
        target_dir / "Flicker8k_Dataset",
        target_dir / "Flickr8k_Dataset",
    ]

    for p_dir in possible_img_dirs:
        if p_dir.exists():
            for img_file in p_dir.glob("*.jpg"):
                dest = images_dir / img_file.name
                if not dest.exists():
                    shutil.move(str(img_file), str(dest))
            shutil.rmtree(p_dir, ignore_errors=True)

    # Copy standard split files
    split_mappings = [
        ("Flickr_8k.trainImages.txt", "train_images.txt"),
        ("Flickr_8k.devImages.txt", "val_images.txt"),
        ("Flickr_8k.testImages.txt", "test_images.txt"),
    ]

    for orig_name, alias_name in split_mappings:
        src_file = target_dir / orig_name
        if src_file.exists():
            shutil.copy(str(src_file), str(splits_dir / orig_name))
            shutil.copy(str(src_file), str(splits_dir / alias_name))

    # Parse raw token captions into standardized captions.txt
    token_file = target_dir / "Flickr8k.token.txt"
    captions_csv = target_dir / "captions.txt"

    if token_file.exists():
        logger.info(f"Generating standardized CSV {captions_csv} from {token_file} ...")
        rows = ["image,caption"]
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1) if "\t" in line else line.split(" ", 1)
                if len(parts) == 2:
                    img_raw, caption = parts[0].strip(), parts[1].strip()
                    img_name = img_raw.split("#")[0].strip()
                    clean_caption = caption.replace('"', '""')
                    rows.append(f'"{img_name}","{clean_caption}"')

        with open(captions_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        logger.info(f"Saved {len(rows)-1} captions into {captions_csv}")


def verify_dataset(target_dir: Path) -> Dict[str, Any]:
    """
    Perform dataset inspection and print results.
    """
    images_dir = target_dir / "Images"
    captions_file = target_dir / "captions.txt"
    splits_dir = target_dir / "splits"

    all_images = list(images_dir.glob("*.jpg"))
    
    # Read captions
    image_to_captions: Dict[str, List[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"'):
                parts = line.split('","')
                if len(parts) == 2:
                    img = parts[0].replace('"', '').strip()
                    cap = parts[1].replace('"', '').strip()
                    image_to_captions.setdefault(img, []).append(cap)
            else:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    img, cap = parts[0].strip(), parts[1].strip()
                    image_to_captions.setdefault(img, []).append(cap)

    # Read splits
    def read_split(filename: str) -> List[str]:
        path = splits_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return []

    train_imgs = read_split("Flickr_8k.trainImages.txt")
    dev_imgs = read_split("Flickr_8k.devImages.txt")
    test_imgs = read_split("Flickr_8k.testImages.txt")

    total_captions = sum(len(c) for c in image_to_captions.values())

    print("\n" + "=" * 65)
    print("      FLICKR8K DATASET VERIFICATION & INSPECTION REPORT")
    print("=" * 65)
    print(f"[*] Total Real Images in {images_dir.name}/:   {len(all_images):,} images")
    print(f"[*] Total Captions in {captions_file.name}:      {total_captions:,} captions")
    print(f"[*] Average Captions per Image:         {total_captions / max(1, len(image_to_captions)):.2f}")
    print(f"[*] Standard Karpathy / Flickr8k Splits:")
    print(f"    - Train Set (trainImages.txt):      {len(train_imgs):,} images ({len(train_imgs)*5:,} captions)")
    print(f"    - Dev/Val Set (devImages.txt):      {len(dev_imgs):,} images ({len(dev_imgs)*5:,} captions)")
    print(f"    - Test Set (testImages.txt):        {len(test_imgs):,} images ({len(test_imgs)*5:,} captions)")
    print("=" * 65)

    print("\n--- SAMPLE IMAGE-CAPTION PAIRS FROM FLICKR8K ---")
    sample_imgs = list(image_to_captions.keys())[:3]
    for i, img in enumerate(sample_imgs, 1):
        print(f"\n[Sample #{i}] Image: {img}")
        for j, cap in enumerate(image_to_captions[img], 1):
            print(f"  ({j}) \"{cap}\"")
    print("=" * 65 + "\n")

    return {
        "num_images": len(all_images),
        "num_captions": total_captions,
        "train_count": len(train_imgs),
        "dev_count": len(dev_imgs),
        "test_count": len(test_imgs),
    }


def main():
    target_dir = Config.paths.DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    text_zip = target_dir / "Flickr8k_text.zip"
    dataset_zip = target_dir / "Flickr8k_Dataset.zip"

    # Step 1 & 2: Download & Extract Text
    logger.info("Step 1/4: Downloading Flickr8k Text Annotations...")
    download_file_multithreaded(TEXT_URL, text_zip, num_threads=4)
    logger.info("Step 2/4: Extracting Flickr8k Text Annotations...")
    extract_zip(text_zip, target_dir)

    # Step 3 & 4: Download & Extract Images (12 parallel threads)
    logger.info("Step 3/4: Downloading Flickr8k Images Dataset (1.04 GB) with 12 parallel threads...")
    download_file_multithreaded(DATASET_URL, dataset_zip, num_threads=12)
    logger.info("Step 4/4: Extracting Flickr8k Images Dataset...")
    extract_zip(dataset_zip, target_dir)

    # Organize files into standard structure
    organize_flickr8k(target_dir)

    # Verify and inspect
    verify_dataset(target_dir)

    # Clean up zip archives
    if text_zip.exists():
        text_zip.unlink()
    if dataset_zip.exists():
        dataset_zip.unlink()
    logger.info("Cleaned up temporary zip files. Flickr8k is ready!")


if __name__ == "__main__":
    main()
