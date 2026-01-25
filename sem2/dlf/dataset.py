"""
Dataset utilities for cross-modal verification
Handles loading Flickr8k/COCO datasets and generating hard negatives
"""
import os
import csv
import json
import random
from functools import lru_cache
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import Counter


class Flickr8kDataset(Dataset):
    """Dataset class for Flickr8k with hard negative generation"""
    
    def __init__(self, image_dir, captions_file, transform=None, 
                 negative_ratio=0.5, hard_negative_ratio=0.7, 
                 vocab=None, max_length=50, mode='train'):
        """
        Args:
            image_dir: Directory containing images
            captions_file: Path to captions file (JSON or text format)
            transform: Image transformations
            negative_ratio: Ratio of negative examples to generate
            hard_negative_ratio: Ratio of hard negatives among all negatives
            vocab: Word vocabulary (will be built if None)
            max_length: Maximum sequence length for text
            mode: 'train' or 'val'
        """
        self.image_dir = image_dir
        self.transform = transform or self.default_transform()
        self.negative_ratio = negative_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.max_length = max_length
        self.mode = mode
        self._token_cache = {}
        
        # Load captions
        self.data = self.load_captions(captions_file)
        
        # Build or use vocabulary
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
            
        # Generate training examples (positive + negative)
        self.examples = self.generate_examples()

        # Pre-tokenize captions to avoid repeated Python string splitting in __getitem__
        # (helps a bit; biggest wins are usually DataLoader workers + avoiding GPU sync)
        for ex in self.examples:
            cap = ex.get('caption')
            if cap is not None and cap not in self._token_cache:
                self._token_cache[cap] = self.tokenize(cap)
        
    def default_transform(self):
        """Default image transformation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_captions(self, captions_file):
        """Load captions from file"""
        data = {}
        
        # Try JSON format first
        if captions_file.endswith('.json'):
            with open(captions_file, 'r') as f:
                raw_data = json.load(f)
                # Assume format: {"image_id": ["caption1", "caption2", ...]}
                for img_id, captions in raw_data.items():
                    data[img_id] = captions if isinstance(captions, list) else [captions]
        else:
            # Text formats supported:
            # - Flickr8k.token.txt: image.jpg#0\tcaption
            # - Kaggle captions.txt: CSV with header "image,caption"
            with open(captions_file, 'r', encoding='utf-8', newline='') as f:
                first_line = f.readline()
                if not first_line:
                    return data
                f.seek(0)

                is_token_format = ('\t' in first_line)
                is_csv_format = (',' in first_line and not is_token_format)

                if is_token_format:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split('\t', 1)
                        if len(parts) < 2:
                            continue

                        img_caption_id, caption = parts[0].strip(), parts[1].strip()
                        img_id = img_caption_id.split('#')[0] if '#' in img_caption_id else img_caption_id
                        if not img_id or not caption:
                            continue
                        data.setdefault(img_id, []).append(caption)

                elif is_csv_format:
                    reader = csv.reader(f)
                    for row in reader:
                        if not row:
                            continue
                        if row[0].strip().lower() == 'image':
                            continue
                        if len(row) < 2:
                            continue
                        img_id = row[0].strip()
                        caption = ','.join(row[1:]).strip()
                        if not img_id or not caption:
                            continue
                        data.setdefault(img_id, []).append(caption)

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
                        data.setdefault(img_id, []).append(caption)
        
        return data
    
    def build_vocab(self, min_freq=2):
        """Build vocabulary from captions"""
        word_counts = Counter()
        
        for captions in self.data.values():
            for caption in captions:
                tokens = caption.lower().split()
                word_counts.update(tokens)
        
        # Create vocab: {word: id}
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for word, count in word_counts.items():
            if count >= min_freq:
                vocab[word] = idx
                idx += 1
                
        print(f"Built vocabulary with {len(vocab)} words")
        return vocab
    
    def tokenize(self, text):
        """Tokenize text to indices"""
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, 1) for token in tokens[:self.max_length]]
        
        # Pad
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
            
        return torch.tensor(token_ids, dtype=torch.long)
    
    def generate_hard_negative_caption(self, correct_caption, all_captions):
        """
        Generate hard negative by modifying key attributes
        Strategies:
        1. Replace color words
        2. Replace number words
        3. Replace objects with similar objects
        4. Use caption from similar image
        """
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'orange', 'pink']
        numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                   '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        tokens = correct_caption.lower().split()
        
        # Strategy 1: Replace colors
        for i, token in enumerate(tokens):
            if token in colors:
                other_colors = [c for c in colors if c != token]
                tokens[i] = random.choice(other_colors)
                return ' '.join(tokens)
        
        # Strategy 2: Replace numbers
        for i, token in enumerate(tokens):
            if token in numbers:
                other_numbers = [n for n in numbers if n != token]
                tokens[i] = random.choice(other_numbers)
                return ' '.join(tokens)
        
        # Strategy 3: Use a different caption (similar style)
        return random.choice(all_captions)
    
    def generate_examples(self):
        """Generate positive and negative examples"""
        examples = []
        all_image_ids = list(self.data.keys())
        all_captions = []
        for caps in self.data.values():
            all_captions.extend(caps)
        
        # Positive examples
        for img_id, captions in self.data.items():
            for caption in captions:
                examples.append({
                    'image_id': img_id,
                    'caption': caption,
                    'label': 1
                })
        
        num_positives = len(examples)
        num_negatives = int(num_positives * self.negative_ratio)
        num_hard_negatives = int(num_negatives * self.hard_negative_ratio)
        num_easy_negatives = num_negatives - num_hard_negatives
        
        # Hard negative examples
        for _ in range(num_hard_negatives):
            img_id = random.choice(all_image_ids)
            correct_caption = random.choice(self.data[img_id])
            hard_negative = self.generate_hard_negative_caption(correct_caption, all_captions)
            examples.append({
                'image_id': img_id,
                'caption': hard_negative,
                'label': 0
            })
        
        # Easy negative examples (completely random pairing)
        for _ in range(num_easy_negatives):
            img_id = random.choice(all_image_ids)
            # Select caption from different image
            other_imgs = [i for i in all_image_ids if i != img_id]
            other_img = random.choice(other_imgs)
            wrong_caption = random.choice(self.data[other_img])
            examples.append({
                'image_id': img_id,
                'caption': wrong_caption,
                'label': 0
            })
        
        # Shuffle examples
        random.shuffle(examples)
        
        print(f"Generated {len(examples)} examples "
              f"({num_positives} positive, {num_negatives} negative)")
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, example['image_id'])
        if not img_path.endswith(('.jpg', '.jpeg', '.png')):
            img_path += '.jpg'  # Default extension
            
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = torch.zeros(3, 224, 224)
        
        # Tokenize caption
        caption = example['caption']
        text_tokens = self._token_cache.get(caption)
        if text_tokens is None:
            text_tokens = self.tokenize(caption)
            self._token_cache[caption] = text_tokens
        
        # Label
        label = torch.tensor(example['label'], dtype=torch.float)
        
        return image, text_tokens, label


def create_dataloaders(image_dir, captions_file, batch_size=32, 
                       train_split=0.8, num_workers=4):
    """Create train and validation dataloaders"""
    
    # Load all data first to build vocab
    full_dataset = Flickr8kDataset(
        image_dir=image_dir,
        captions_file=captions_file,
        mode='train'
    )

    if len(full_dataset) == 0:
        raise ValueError(
            "No training examples were generated. "
            "This usually means captions could not be loaded (wrong format/path) "
            "or captions are empty. Ensure you have data/captions.json (or a valid captions.txt) "
            "and that it contains entries." 
        )
    
    # Split into train/val
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split_idx = int(dataset_size * train_split)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Create dataloaders
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers is not None and num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None
    )
    
    return train_loader, val_loader, full_dataset.vocab
