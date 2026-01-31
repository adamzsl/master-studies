"""
SubmissionModel for cross-modal verification system
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import json
import os

# Define defaults here to make model.py self-contained for submission
DEFAULT_VOCAB_SIZE = 10000
DEFAULT_EMBED_DIM = 256
DEFAULT_IMAGE_EMBED_DIM = 512
DEFAULT_HIDDEN_DIM = 256

class ImageEncoder(nn.Module):
    """CNN-based image encoder using pretrained ResNet"""
    
    def __init__(self, embedding_dim=DEFAULT_IMAGE_EMBED_DIM, pretrained=True, backbone='resnet50'):
        super(ImageEncoder, self).__init__()

        # Handle potential backbone strings
        backbone = (backbone or 'resnet50').lower()
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        else:
            resnet = models.resnet50(pretrained=pretrained)

        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add projection layer
        # ResNet18/34 have 512 output features, ResNet50/101 have 2048
        in_features = resnet.fc.in_features
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        features = self.features(x)  # [batch_size, C, 1, 1]
        features = features.view(features.size(0), -1)  # Flatten
        embeddings = self.projection(features)
        return embeddings


class TextEncoder(nn.Module):
    """LSTM-based text encoder"""
    
    def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE, embedding_dim=DEFAULT_EMBED_DIM, 
                 hidden_dim=DEFAULT_HIDDEN_DIM, num_layers=2, bidirectional=True, dropout=0.3):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        # Calculate output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward final states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
            
        embeddings = self.projection(hidden)
        return embeddings


class FusionModule(nn.Module):
    """Fusion module to combine image and text embeddings"""
    
    def __init__(self, embedding_dim=DEFAULT_EMBED_DIM, hidden_dim=DEFAULT_HIDDEN_DIM, dropout=0.3):
        super(FusionModule, self).__init__()
        # We fuse: [image, text, image*text] -> 3x embedding_dim
        fusion_input_dim = embedding_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, image_emb, text_emb):
        # Element-wise multiplication (interactions)
        mult_features = image_emb * text_emb
        
        # Concatenation
        concat_features = torch.cat([image_emb, text_emb], dim=1)
        
        # Combine all
        combined = torch.cat([concat_features, mult_features], dim=1)
            
        logits = self.fusion(combined)
        return logits.squeeze(1)


class SubmissionModel(nn.Module):
    """
    Main model for cross-modal verification
    """
    
    def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE, 
                 image_embedding_dim=DEFAULT_IMAGE_EMBED_DIM, 
                 text_embedding_dim=DEFAULT_EMBED_DIM, 
                 hidden_dim=DEFAULT_HIDDEN_DIM,
                 cnn_backbone='resnet50', pretrained=True):
        super(SubmissionModel, self).__init__()
        
        # 1. Load Vocab immediately (Critical for submission server)
        self.vocab = {}
        self.max_length = 50
        loaded_vocab_size = self.load_vocab_from_file()
        actual_vocab_size = loaded_vocab_size or vocab_size
        self.vocab_size = actual_vocab_size

        self.image_encoder = ImageEncoder(
            embedding_dim=image_embedding_dim,
            pretrained=pretrained,
            backbone=cnn_backbone
        )
        
        self.text_encoder = TextEncoder(
            vocab_size=actual_vocab_size,
            embedding_dim=text_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        # Project image features to match text embedding size for fusion
        self.image_projection = nn.Linear(image_embedding_dim, text_embedding_dim)
        
        self.fusion = FusionModule(
            embedding_dim=text_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=0.3
        )
        
    def load_vocab_from_file(self, filename='vocab.json'):
        """Attempt to load vocabulary from json file. Returns vocab size if loaded."""
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, 'r') as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                return None
            # Ensure ints
            self.vocab = {str(k): int(v) for k, v in loaded.items()}
            if not self.vocab:
                return None
            max_idx = max(self.vocab.values())
            return int(max_idx) + 1
        except Exception as e:
            print(f"Warning: Could not load vocab.json: {e}")
            return None

    def forward(self, images, texts):
        # Encode image
        image_features = self.image_encoder(images)
        image_features = self.image_projection(image_features) # Project to 256
        
        # Encode text
        text_features = self.text_encoder(texts)
        
        # Fuse
        output = self.fusion(image_features, text_features)
        return output
    
    def tokenize(self, text):
        """Simple word-level tokenization using loaded vocab"""
        tokens = text.lower().split()
        # Default to 1 (UNK) if word not found. 0 is PAD.
        token_ids = [self.vocab.get(token, 1) for token in tokens[:self.max_length]]
        
        # Pad to max_length
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
            
        return torch.tensor(token_ids, dtype=torch.long)
    
    def predict(self, image_tensor, text_string):
        """
        Required method for submission.
        """
        self.eval()
        with torch.no_grad():
            # 1. Prepare Image [1, 3, 224, 224]
            image = image_tensor.unsqueeze(0)
            
            # 2. Prepare Text [1, 50]
            text_tokens = self.tokenize(text_string).unsqueeze(0)
            
            # 3. Device handling
            device = next(self.parameters()).device
            image = image.to(device)
            text_tokens = text_tokens.to(device)
            
            # 4. Forward
            logits = self.forward(image, text_tokens)
            
            # 5. Return probability float
            return torch.sigmoid(logits).item()
    
    def set_vocab(self, vocab):
        """Helper to set vocab manually if needed"""
        self.vocab = vocab


def get_transform():
    """
    Required transform function
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])