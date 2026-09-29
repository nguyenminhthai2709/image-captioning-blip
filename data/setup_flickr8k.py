"""
Helper script to prepare, organize, and download Flickr8k dataset,
or generate a synthetic mock subset for local pipeline testing & demonstration.
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Config
from src.utils import setup_logger

logger = setup_logger("DatasetSetup")


def create_sample_dataset(target_dir: Path, num_samples: int = 25) -> None:
    """
    Generate synthetic sample images and multi-caption annotations
    for immediate end-to-end pipeline execution and testing.
    """
    images_dir = target_dir / "Images"
    splits_dir = target_dir / "splits"
    images_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    captions_file = target_dir / "captions.txt"
    
    scenarios = [
        ("dog_running.jpg", [
            "A brown dog running happily through the green grass",
            "A canine running across an open lawn",
            "A golden retriever sprinting in the park",
            "A dog chasing a ball in a field",
            "A happy pet playing on a grassy field"
        ]),
        ("children_playing.jpg", [
            "Two children playing with colorful blocks on the floor",
            "A boy and a girl building a toy tower",
            "Toddlers playing together in a living room",
            "Kids sitting on a rug and playing with toys",
            "Two little kids playing with wooden toy blocks"
        ]),
        ("cat_window.jpg", [
            "A fluffy white cat sitting by a sunny window",
            "A cat gazing outside through the glass window",
            "A white feline relaxing on the window ledge in sunlight",
            "A pet cat looking out at the birds outside",
            "A domestic cat looking outside on a sunny afternoon"
        ]),
        ("cyclist_mountain.jpg", [
            "A cyclist riding a bicycle along a winding mountain trail",
            "A person in a helmet biking on a dirt path in the hills",
            "A mountain biker riding through scenic outdoor scenery",
            "A cyclist pedaling on a trail surrounded by mountains",
            "An athletic person riding a bike on a trail"
        ]),
        ("surfer_wave.jpg", [
            "A surfer riding a massive ocean wave in the sunshine",
            "A person on a surfboard balancing on blue ocean water",
            "A young surfer catching a barrel wave at the beach",
            "An athlete surfing in clear blue sea water",
            "A surfer wearing a wetsuit riding a cresting wave"
        ]),
        ("chef_cooking.jpg", [
            "A professional chef preparing fresh food in a restaurant kitchen",
            "A person wearing an apron slicing vegetables on a board",
            "A cook stirring ingredients in a large pan on the stove",
            "A chef garnishing a gourmet dinner plate",
            "A cook making a healthy meal in the kitchen"
        ]),
        ("sunset_beach.jpg", [
            "A peaceful sunset over a sandy beach with calm waves",
            "The golden sun setting over the ocean horizon",
            "Warm orange sky reflected on the ocean at twilight",
            "People walking along the shore during a beautiful sunset",
            "A tropical beach glowing in the evening sunlight"
        ]),
        ("girl_reading.jpg", [
            "A young girl reading an open book in a library",
            "A student sitting at a wooden desk with books",
            "A girl with glasses studying quietly in the library",
            "A child absorbed in reading a storybook",
            "A young person looking through pages of an old book"
        ]),
        ("guitar_player.jpg", [
            "A street musician playing an acoustic guitar on a sidewalk",
            "A man strumming guitar chords outdoors for pedestrians",
            "An artist performing music with a guitar in a city square",
            "A musician playing an acoustic instrument in the street",
            "A person holding a wooden guitar and singing"
        ]),
        ("airplane_clouds.jpg", [
            "A commercial airplane flying high above fluffy white clouds",
            "A jet aircraft soaring through the bright blue sky",
            "An airplane wing visible above cloud formations",
            "A passenger jet cruising at high altitude in the sky",
            "A plane flying peacefully above the sea of clouds"
        ])
    ]
    
    # Expand to requested number of samples
    all_rows = []
    image_names = []
    
    colors = [
        (41, 128, 185), (39, 174, 96), (230, 126, 34), (142, 68, 173),
        (192, 57, 43), (22, 160, 133), (243, 156, 18), (52, 73, 94)
    ]
    
    for i in range(num_samples):
        base_name, caps = scenarios[i % len(scenarios)]
        img_name = f"{i+1:04d}_{base_name}"
        image_names.append(img_name)
        
        # Create synthetic image with geometric patterns and labels
        img = Image.new("RGB", (384, 384), color=colors[i % len(colors)])
        draw = ImageDraw.Draw(img)
        
        # Draw background shapes for visual variety
        draw.ellipse([40, 40, 340, 340], fill=(255, 255, 255, 50), outline=(240, 240, 240))
        draw.rectangle([80, 120, 300, 260], fill=(245, 245, 245), outline=(200, 200, 200))
        
        # Draw title text
        tag_text = base_name.replace(".jpg", "").replace("_", " ").upper()
        draw.text((100, 180), f"SAMPLE {i+1}:\n{tag_text}", fill=(30, 30, 30))
        
        img.save(images_dir / img_name, format="JPEG", quality=90)
        
        # Add 5 captions
        for cap in caps:
            all_rows.append(f"{img_name},{cap}")
            
    # Write captions.txt in standard CSV format
    with open(captions_file, "w", encoding="utf-8") as f:
        f.write("image,caption\n")
        for row in all_rows:
            f.write(f"{row}\n")
            
    # Write Train / Val / Test Splits (60% train, 20% val, 20% test)
    n_train = int(len(image_names) * 0.6)
    n_val = int(len(image_names) * 0.2)
    
    train_imgs = image_names[:n_train]
    val_imgs = image_names[n_train:n_train + n_val]
    test_imgs = image_names[n_train + n_val:]
    
    with open(splits_dir / "train_images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_imgs))
    with open(splits_dir / "val_images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_imgs))
    with open(splits_dir / "test_images.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_imgs))
        
    logger.info(
        f"Generated sample dataset at {target_dir}:\n"
        f"- Total Images: {len(image_names)}\n"
        f"- Total Captions: {len(all_rows)}\n"
        f"- Splits: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}"
    )


if __name__ == "__main__":
    Config.create_dirs()
    create_sample_dataset(Config.paths.DATA_DIR, num_samples=20)
