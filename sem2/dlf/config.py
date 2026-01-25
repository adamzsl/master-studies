"""
Configuration file for the cross-modal verification system
"""
import torch

# Model Configuration
IMAGE_EMBEDDING_DIM = 512
TEXT_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
DROPOUT = 0.3

# CNN Configuration
# Jeśli trening jest wolny (szczególnie na CPU), ustaw np. 'resnet18'.
CNN_BACKBONE = 'resnet50'  # resnet18, resnet34, resnet50
PRETRAINED = True

# LSTM Configuration
LSTM_LAYERS = 2
BIDIRECTIONAL = True
MAX_TEXT_LENGTH = 50

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-5

# Data Configuration
IMAGE_SIZE = 224
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR = './data'
CHECKPOINT_DIR = './checkpoints'
WEIGHTS_PATH = './weights.pth'

# Negative Examples Strategy
NEGATIVE_RATIO = 0.5  # Ratio of negative to positive examples
HARD_NEGATIVE_RATIO = 0.7  # Ratio of hard negatives among all negatives
