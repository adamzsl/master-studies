# Quick Reference - CNN+LSTM Cross-Modal Verification

## 🚀 Quick Start

```bash
cd sem2/dlf

# Install dependencies
pip install -r requirements.txt

# Prepare data (create test dataset)
python prepare_data.py data

# Train model
python train.py

# Test submission format
python test_submission.py

# Create submission
zip submission.zip model.py weights.pth vocab.json
```

## 📁 File Structure

```
sem2/dlf/
├── model.py              # SubmissionModel (MAIN FILE)
├── weights.pth           # Trained weights (MAIN FILE)
├── vocab.json            # Vocabulary (generated)
├── config.py             # Hyperparameters
├── dataset.py            # Data loading + hard negatives
├── train.py              # Training script
├── prepare_data.py       # Data preparation
├── test_submission.py    # Submission validator
├── test_model.py         # Unit tests
├── train.sh              # Quick start script
├── colab_training.py     # Google Colab script
├── README.md            # Full documentation
├── RAPORT.md            # Report template
├── KROK_PO_KROKU.md     # Step-by-step guide (Polish)
└── QUICK_REFERENCE.md   # This file
```

## 🏗️ Architecture

```
Image (224×224) ──> ResNet-50 ──> CNN Features (2048) ──> Projection (256)
                                                               │
                                                               ↓
Text (words) ──> Embeddings ──> Bi-LSTM (256×2) ──> Projection (256)
                                                               │
                                                               ↓
                          [Concat + Multiply] ──> MLP ──> Sigmoid(1)
```

## ⚙️ Key Hyperparameters

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
IMAGE_EMBEDDING_DIM = 512
TEXT_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
DROPOUT = 0.3
```

## 📊 Data Requirements

### Structure
```
data/
├── images/
│   └── *.jpg
└── captions.json
```

### Captions Format
```json
{
  "image1.jpg": ["caption 1", "caption 2", ...],
  "image2.jpg": ["caption 1", "caption 2", ...]
}
```

## 🎯 Hard Negatives Strategy

### 70% Hard Negatives
- Color swap: "red car" → "blue car"
- Number swap: "three dogs" → "two dogs"
- Object swap: similar objects

### 30% Easy Negatives
- Random mismatched pairs

## 📈 Training Process

1. Load data (80% train, 20% val)
2. Build vocabulary
3. Generate positives + negatives
4. Train with BCELoss
5. Validate each epoch
6. Save best model
7. Generate training curves

## 🧪 Testing Commands

```bash
# Unit tests
python test_model.py

# Submission validation
python test_submission.py

# Quick model test
python -c "
from model import SubmissionModel
import torch
model = SubmissionModel()
img = torch.randn(3, 224, 224)
txt = 'test caption'
vocab = {'test': 2, 'caption': 3}
model.set_vocab(vocab)
print(f'Score: {model.predict(img, txt):.4f}')
"
```

## 📦 Submission Files

### Required
1. `model.py` - Must contain `SubmissionModel` class
2. `weights.pth` - Trained model weights

### Optional
- `vocab.json` - Vocabulary dictionary
- `config.py` - Configuration
- Other custom modules

### SubmissionModel Requirements
```python
class SubmissionModel(nn.Module):
    def predict(self, image_tensor, text_string):
        """
        Args:
            image_tensor: torch.Tensor [3, 224, 224]
            text_string: str
        Returns:
            float: 0.0 - 1.0 (use .item()!)
        """
```

## 🎓 Grading Thresholds

| Grade | Requirement |
|-------|-------------|
| 3.0 | >65% (test set 1) |
| 3.5 | >70% (test set 1) |
| 4.0 | >75% (test sets 1, 2) |
| 4.5 | >80% (test sets 1, 2, 3) |
| 5.0 | >85% (test sets 1, 2, 3) |

*+5% for Transformer models

## 🐛 Common Issues & Fixes

### Low Accuracy (~50%)
- Increase hard negatives ratio
- Train longer (50+ epochs)
- Check class balance
- Verify fusion layer

### Overfitting
- Increase dropout (0.3 → 0.5)
- Add weight decay (1e-4)
- More training data
- Early stopping

### Out of Memory
- Reduce batch size (32 → 16)
- Smaller model (ResNet-34)
- Gradient accumulation

### Model Not Learning
- Check data loading
- Verify loss calculation
- Lower learning rate
- Check gradients

## 📊 Monitoring

### Files Generated
- `training_curves.png` - Loss & accuracy plots
- `checkpoints/best_model.pth` - Best model
- `checkpoints/checkpoint_epoch_X.pth` - Periodic saves

### Metrics Tracked
- Loss (train, val)
- Accuracy (main metric)
- Precision
- Recall
- F1-Score

## 💡 Tips for Better Performance

### Data
✓ Use COCO instead of Flickr8k (more data)
✓ Implement more hard negative strategies
✓ Data augmentation (rotation, crop, color jitter)

### Model
✓ Larger backbone (ResNet-101)
✓ Deeper LSTM (3-4 layers)
✓ Attention mechanism in fusion
✓ Multiple fusion strategies

### Training
✓ Longer training (50+ epochs)
✓ Learning rate scheduling
✓ Gradient clipping
✓ Mixed precision training

### Ensemble
✓ Train multiple models
✓ Different architectures
✓ Average predictions

## 🔗 Useful Links

- **Flickr8k**: https://www.kaggle.com/datasets/adityajn105/flickr8k
- **COCO**: https://cocodataset.org/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Submission System**: [URL from course materials]

## 📝 Before Submission Checklist

- [ ] Model trains without errors
- [ ] Validation accuracy >70%
- [ ] All unit tests pass (11/11)
- [ ] `test_submission.py` passes all checks
- [ ] `submission.zip` created
- [ ] File size <100MB
- [ ] Report filled out (RAPORT.md)

## 🆘 Getting Help

1. Check `KROK_PO_KROKU.md` for detailed guide
2. Check `README.md` for full documentation
3. Run `test_model.py` for validation
4. Check training curves for issues
5. Review example predictions

## ⚡ Speed Tips

### Fast Testing
```bash
# Use minimal dataset
python prepare_data.py data  # Choose 'y' for minimal

# Reduce epochs for testing
# Edit config.py: NUM_EPOCHS = 2
```

### Google Colab
```bash
# Copy colab_training.py to Colab
# Free GPU (Tesla T4)
# ~15-20 min per epoch
```

### Local GPU
```bash
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Training time (with GPU):
# - Flickr8k: ~5-10 min/epoch
# - COCO: ~30-60 min/epoch
```

## 📊 Expected Results

### Baseline
- Random: ~50%
- Simple CNN+LSTM: 65-70%

### With Hard Negatives
- Good implementation: 75-80%
- Excellent implementation: 80-85%

### State-of-art
- CLIP (Transformer): 85-90%

---

**Model Parameters**: 27,935,297  
**Model Size**: ~107 MB  
**Min RAM**: 8 GB  
**Min VRAM**: 4 GB  
**Training Time**: 20 epochs × 10 min = ~3-4 hours (on GPU)

---

**Last Updated**: January 2025  
**Status**: ✅ Complete & Ready  
**Tests**: ✅ 11/11 Passing
