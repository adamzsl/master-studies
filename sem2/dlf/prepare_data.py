"""
Utility script to prepare Flickr8k dataset for training
Downloads and formats Flickr8k dataset if needed
"""
import os
import csv
import json
import zipfile
import requests
from tqdm import tqdm
from PIL import Image


def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def extract_flickr8k_captions(captions_file, output_file):
    """
    Extract captions from Flickr8k caption files.

    Supported input formats:
    - Flickr8k.token.txt: image_id.jpg#0\tcaption
    - Kaggle captions.txt: CSV with header "image,caption"
    Output format: JSON with {image_id: [captions]}
    """
    captions_dict = {}

    with open(captions_file, 'r', encoding='utf-8', newline='') as f:
        # Detect format (tab-delimited token file vs CSV)
        first_line = f.readline()
        if not first_line:
            return captions_dict
        f.seek(0)

        is_token_format = ('\t' in first_line)
        is_csv_format = (',' in first_line and not is_token_format)

        if is_token_format:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                img_caption_id = parts[0].strip()
                caption = parts[1].strip()

                # Extract image id (remove #0, #1, etc.)
                img_id = img_caption_id.split('#')[0] if '#' in img_caption_id else img_caption_id
                if not img_id:
                    continue

                captions_dict.setdefault(img_id, []).append(caption)

        elif is_csv_format:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue

                # Skip header if present
                if row[0].strip().lower() == 'image':
                    continue

                if len(row) < 2:
                    continue

                img_id = row[0].strip()
                caption = ','.join(row[1:]).strip()
                if not img_id or not caption:
                    continue

                captions_dict.setdefault(img_id, []).append(caption)

        else:
            # Fallback: try tab split first, then comma split
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if '\t' in line:
                    parts = line.split('\t', 1)
                    if len(parts) < 2:
                        continue
                    img_caption_id, caption = parts[0].strip(), parts[1].strip()
                    img_id = img_caption_id.split('#')[0] if '#' in img_caption_id else img_caption_id
                elif ',' in line:
                    parts = line.split(',', 1)
                    if len(parts) < 2:
                        continue
                    img_id, caption = parts[0].strip(), parts[1].strip()
                    if img_id.lower() == 'image':
                        continue
                else:
                    continue

                if not img_id or not caption:
                    continue
                captions_dict.setdefault(img_id, []).append(caption)
    
    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(captions_dict, f, indent=2)
    
    print(f"Extracted captions for {len(captions_dict)} images")
    print(f"Saved to {output_file}")
    
    return captions_dict


def prepare_flickr8k(data_dir='./data'):
    """
    Prepare Flickr8k dataset
    
    Expected structure after preparation:
    data/
            Images/
        image1.jpg
        image2.jpg
        ...
      captions.json
    """
    print("Preparing Flickr8k dataset...")
    
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, 'Images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Check if captions file exists
    captions_file = None
    for filename in ['Flickr8k.token.txt', 'captions.txt', 'Flickr_8k.trainImages.txt']:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            captions_file = filepath
            break
    
    if captions_file is None:
        print("\nFlickr8k caption file not found!")
        print("Please download Flickr8k dataset from:")
        print("https://www.kaggle.com/datasets/adityajn105/flickr8k")
        print("\nOr use the following structure:")
        print("data/")
        print("  Flickr8k.token.txt  (captions file)")
        print("  Flicker8k_Dataset/  (images directory)")
        print("\nAlternatively, you can use COCO dataset.")
        return False
    
    # Extract and convert captions
    output_captions = os.path.join(data_dir, 'captions.json')
    if not os.path.exists(output_captions):
        print(f"\nExtracting captions from {captions_file}...")
        captions_dict = extract_flickr8k_captions(captions_file, output_captions)
    else:
        print(f"Captions file already exists: {output_captions}")
        with open(output_captions, 'r') as f:
            captions_dict = json.load(f)
    
    if not captions_dict:
        print("\nNo captions were extracted.")
        print("This usually means the captions file format wasn't recognized or the file is empty.")
        print(f"Checked file: {captions_file}")
        print("Expected either Flickr8k.token.txt (tab) or captions.txt (CSV with 'image,caption').")
        return False

    # Check if images are in the right place
    sample_img = next(iter(captions_dict.keys()))
    sample_path = os.path.join(images_dir, sample_img)
    
    if not os.path.exists(sample_path):
        # Try to find images in subdirectory
        alt_dirs = ['Flicker8k_Dataset', 'Flickr8k_Dataset', 'Images', 'images', 'Image']
        for alt_dir in alt_dirs:
            alt_path = os.path.join(data_dir, alt_dir, sample_img)
            if os.path.exists(alt_path):
                print(f"\nImages found in {alt_dir}/")
                print(f"Please copy or move images to {images_dir}/")
                print(f"Example: cp {os.path.join(data_dir, alt_dir)}/* {images_dir}/")
                return False
        
        print(f"\nImages not found!")
        print(f"Please place images in: {images_dir}/")
        return False
    
    print("\n✓ Dataset prepared successfully!")
    print(f"  Images: {images_dir}/")
    print(f"  Captions: {output_captions}")
    print(f"  Total images: {len(captions_dict)}")
    
    return True


def create_minimal_dataset(data_dir='./data', num_samples=50):
    """
    Create a minimal dataset for testing
    Useful when you don't have access to full dataset
    """
    print(f"Creating minimal test dataset with {num_samples} samples...")
    
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, 'Images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Sample captions
    sample_captions = [
        "a dog playing in the park",
        "a cat sitting on a couch",
        "a person riding a bicycle",
        "a red car parked on the street",
        "children playing soccer",
        "a bird flying in the sky",
        "a woman walking her dog",
        "a man reading a book",
        "a group of people at a beach",
        "a train at the station",
    ]
    
    captions_dict = {}
    
    # Create dummy images and captions
    for i in range(num_samples):
        img_name = f"image{i:04d}.jpg"
        img_path = os.path.join(images_dir, img_name)
        
        # Create a random colored image
        img = Image.new('RGB', (224, 224), 
                       color=(i*5 % 256, i*7 % 256, i*11 % 256))
        img.save(img_path)
        
        # Assign random captions
        captions_dict[img_name] = [
            sample_captions[i % len(sample_captions)],
            sample_captions[(i+1) % len(sample_captions)],
        ]
    
    # Save captions
    captions_file = os.path.join(data_dir, 'captions.json')
    with open(captions_file, 'w') as f:
        json.dump(captions_dict, f, indent=2)
    
    print(f"\n✓ Minimal dataset created!")
    print(f"  Images: {images_dir}/")
    print(f"  Captions: {captions_file}")
    print(f"  Total samples: {num_samples}")
    print("\nNOTE: This is a dummy dataset for testing only.")
    print("Please use real Flickr8k or COCO data for actual training.")


if __name__ == '__main__':
    import sys
    
    data_dir = './data'
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    # Try to prepare real dataset
    success = prepare_flickr8k(data_dir)
    
    # If failed, offer to create minimal dataset
    if not success:
        response = input("\nCreate minimal test dataset instead? (y/n): ")
        if response.lower() == 'y':
            create_minimal_dataset(data_dir)
