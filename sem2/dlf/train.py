"""
Training script for cross-modal verification model
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from model import SubmissionModel
from dataset import Flickr8kDataset, create_dataloaders
from config import *


class Trainer:
    """Training class for the model"""
    
    def __init__(self, model, train_loader, val_loader, vocab, 
                 device=DEVICE, learning_rate=LEARNING_RATE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=2,
            verbose=True
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0

        # AMP (mixed precision) speeds up CUDA training significantly
        self.use_amp = (str(self.device).startswith('cuda') or getattr(self.device, 'type', '') == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @staticmethod
    def _binary_metrics(preds: torch.Tensor, labels: torch.Tensor):
        """Compute accuracy/precision/recall/f1 for binary preds/labels on-device."""
        preds_b = preds.bool()
        labels_b = labels.bool()

        tp = (preds_b & labels_b).sum()
        tn = ((~preds_b) & (~labels_b)).sum()
        fp = (preds_b & (~labels_b)).sum()
        fn = ((~preds_b) & labels_b).sum()

        total = tp + tn + fp + fn
        acc = (tp + tn).float() / total.clamp_min(1)
        prec = tp.float() / (tp + fp).clamp_min(1)
        rec = tp.float() / (tp + fn).clamp_min(1)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
        return acc, prec, rec, f1
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        tp = torch.zeros((), device=self.device)
        tn = torch.zeros((), device=self.device)
        fp = torch.zeros((), device=self.device)
        fn = torch.zeros((), device=self.device)
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, texts, labels in pbar:
            images = images.to(self.device)
            texts = texts.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images, texts)
                loss = self.criterion(outputs, labels)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            preds = (outputs > 0.5).float()
            preds_b = preds.bool()
            labels_b = labels.bool()
            tp += (preds_b & labels_b).sum()
            tn += ((~preds_b) & (~labels_b)).sum()
            fp += (preds_b & (~labels_b)).sum()
            fn += ((~preds_b) & labels_b).sum()

            total_loss += loss.detach()
            
            # Update progress bar
            pbar.set_postfix({'loss': float(loss.detach().cpu())})
        
        avg_loss = (total_loss / max(1, len(self.train_loader))).item()
        total = (tp + tn + fp + fn).clamp_min(1)
        accuracy = ((tp + tn).float() / total).item()
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        tp = torch.zeros((), device=self.device)
        tn = torch.zeros((), device=self.device)
        fp = torch.zeros((), device=self.device)
        fn = torch.zeros((), device=self.device)
        
        with torch.no_grad():
            for images, texts, labels in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images, texts)
                    loss = self.criterion(outputs, labels)
                
                # Metrics
                preds = (outputs > 0.5).float()
                preds_b = preds.bool()
                labels_b = labels.bool()
                tp += (preds_b & labels_b).sum()
                tn += ((~preds_b) & (~labels_b)).sum()
                fp += (preds_b & (~labels_b)).sum()
                fn += ((~preds_b) & labels_b).sum()

                total_loss += loss.detach()
        
        avg_loss = (total_loss / max(1, len(self.val_loader))).item()
        total = (tp + tn + fp + fn).clamp_min(1)
        accuracy = ((tp + tn).float() / total).item()
        precision = (tp.float() / (tp + fp).clamp_min(1)).item()
        recall = (tp.float() / (tp + fn).clamp_min(1)).item()
        f1 = (2 * (tp.float() / (tp + fp).clamp_min(1)) * (tp.float() / (tp + fn).clamp_min(1))
              / ((tp.float() / (tp + fp).clamp_min(1)) + (tp.float() / (tp + fn).clamp_min(1))).clamp_min(1e-12)).item()
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self, num_epochs=NUM_EPOCHS):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_prec, val_recall, val_f1 = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_prec:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth')
                print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*50}")
        
        # Plot training curves
        self.plot_training_curves()
        
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        filepath = os.path.join(CHECKPOINT_DIR, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab': self.vocab,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Accuracy', marker='o')
        ax2.plot(self.val_accs, label='Val Accuracy', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("Training curves saved to training_curves.png")
        plt.close()


def prepare_submission_model(checkpoint_path, output_path='weights.pth'):
    """Prepare model for submission"""
    print("Preparing submission model...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    vocab = checkpoint['vocab']
    
    # Create model
    model = SubmissionModel(vocab_size=len(vocab))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_vocab(vocab)
    
    # Save weights
    torch.save(model.state_dict(), output_path)
    print(f"Weights saved to {output_path}")
    
    # Save vocab separately for reference
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    print("Vocabulary saved to vocab.json")
    
    return model, vocab


def main():
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
    
    print("="*50)
    print("Cross-Modal Verification Training")
    print("="*50)
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"\nWARNING: Data directory {DATA_DIR} not found!")
        print("Please download Flickr8k or COCO dataset and update DATA_DIR in config.py")
        print("\nCreating dummy dataset for testing...")
        
        # Create dummy data for testing
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'Images'), exist_ok=True)
        
        # Create dummy captions file
        dummy_captions = {
            'image1.jpg': ['a dog playing in the park', 'a brown dog running'],
            'image2.jpg': ['a red car on the street', 'a vehicle parked outside'],
            'image3.jpg': ['a person riding a bike', 'a cyclist on the road']
        }
        with open(os.path.join(DATA_DIR, 'captions.json'), 'w') as f:
            json.dump(dummy_captions, f)
        
        print("Dummy data created. Please replace with real data for actual training.")
        return
    
    # Create dataloaders
    image_dir = os.path.join(DATA_DIR, 'Images')
    if not os.path.exists(image_dir):
        alt_image_dir = os.path.join(DATA_DIR, 'images')
        if os.path.exists(alt_image_dir):
            image_dir = alt_image_dir
    captions_file = os.path.join(DATA_DIR, 'captions.json')
    
    if not os.path.exists(captions_file):
        # Try alternative caption file names
        for alt_name in ['captions.txt', 'annotations.json', 'captions.token.txt']:
            alt_path = os.path.join(DATA_DIR, alt_name)
            if os.path.exists(alt_path):
                captions_file = alt_path
                break
    else:
        # If captions.json exists but is empty, fall back to captions.txt if available
        try:
            with open(captions_file, 'r') as f:
                maybe_data = json.load(f)
            if isinstance(maybe_data, dict) and len(maybe_data) == 0:
                alt_path = os.path.join(DATA_DIR, 'captions.txt')
                if os.path.exists(alt_path):
                    print("WARNING: captions.json is empty; using captions.txt instead.")
                    captions_file = alt_path
        except Exception:
            pass
    
    print(f"\nLoading data from {DATA_DIR}...")
    train_loader, val_loader, vocab = create_dataloaders(
        image_dir=image_dir,
        captions_file=captions_file,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT
    )
    
    # Create model
    print(f"\nCreating model with vocabulary size: {len(vocab)}")
    model = SubmissionModel(
        vocab_size=len(vocab),
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        device=DEVICE,
        learning_rate=LEARNING_RATE
    )
    
    # Train
    trainer.train(num_epochs=NUM_EPOCHS)
    
    # Prepare submission
    best_checkpoint = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    if os.path.exists(best_checkpoint):
        prepare_submission_model(best_checkpoint, WEIGHTS_PATH)
        print(f"\nSubmission files ready:")
        print(f"  - model.py")
        print(f"  - weights.pth")
        print(f"\nCreate a ZIP file with these files for submission.")
    else:
        print(f"\nNo checkpoint found at {best_checkpoint}")


if __name__ == '__main__':
    main()
