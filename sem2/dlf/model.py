"""
SubmissionModel for cross-modal verification system
Using CNN (ResNet) for image encoding and LSTM for text encoding
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional

from config import CNN_BACKBONE, PRETRAINED


class ImageEncoder(nn.Module):
    """CNN-based image encoder using pretrained ResNet"""
    
    def __init__(self, embedding_dim=512, pretrained=True, backbone: str = 'resnet50'):
        super(ImageEncoder, self).__init__()

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
        in_features = resnet.fc.in_features
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        # x: [batch_size, 3, 224, 224]
        features = self.features(x)  # [batch_size, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [batch_size, 2048]
        embeddings = self.projection(features)  # [batch_size, embedding_dim]
        return embeddings


class TextEncoder(nn.Module):
    """LSTM-based text encoder"""
    
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=256, 
                 num_layers=2, bidirectional=True, dropout=0.3):
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
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state (concatenate both directions if bidirectional)
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, hidden_dim*2]
        else:
            hidden = hidden[-1]  # [batch_size, hidden_dim]
            
        embeddings = self.projection(hidden)  # [batch_size, embedding_dim]
        return embeddings


class FusionModule(nn.Module):
    """Fusion module to combine image and text embeddings"""
    
    def __init__(self, embedding_dim=256, hidden_dim=256, dropout=0.3):
        super(FusionModule, self).__init__()
        # Concatenation-based fusion
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image_emb, text_emb):
        # image_emb: [batch_size, embedding_dim]
        # text_emb: [batch_size, embedding_dim]
        
        # Element-wise multiplication (Hadamard product)
        mult_features = image_emb * text_emb
        
        # Concatenation
        concat_features = torch.cat([image_emb, text_emb], dim=1)
        
        # Combine both
        combined = torch.cat([concat_features, mult_features], dim=1)
        
        # Adjust fusion layer input
        if not hasattr(self, 'fusion_adjusted'):
            self.fusion = nn.Sequential(
                nn.Linear(combined.size(1), 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ).to(combined.device)
            self.fusion_adjusted = True
            
        output = self.fusion(combined)  # [batch_size, 1]
        return output.squeeze(1)  # [batch_size]


class SubmissionModel(nn.Module):
    """
    Main model for cross-modal verification
    Follows submission system requirements
    """
    
    def __init__(self, vocab_size=10000, image_embedding_dim=512, 
                 text_embedding_dim=256, hidden_dim=256,
                 cnn_backbone: str = CNN_BACKBONE, pretrained: bool = PRETRAINED):
        super(SubmissionModel, self).__init__()
        
        self.image_encoder = ImageEncoder(
            embedding_dim=image_embedding_dim,
            pretrained=pretrained,
            backbone=cnn_backbone
        )
        
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        
        # Project to same dimension for fusion
        self.image_projection = nn.Linear(image_embedding_dim, text_embedding_dim)
        
        self.fusion = FusionModule(
            embedding_dim=text_embedding_dim,
            hidden_dim=hidden_dim,
            dropout=0.3
        )
        
        # Simple tokenizer for inference
        self.vocab = {}
        self.max_length = 50
        
    def forward(self, images, texts):
        """Forward pass for training"""
        # Encode image and text
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        
        # Project image features to same dimension
        image_features = self.image_projection(image_features)
        
        # Fuse and predict
        output = self.fusion(image_features, text_features)
        return output
    
    def tokenize(self, text):
        """Simple word-level tokenization"""
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, 1) for token in tokens[:self.max_length]]
        
        # Pad or truncate
        if len(token_ids) < self.max_length:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
            
        return torch.tensor(token_ids, dtype=torch.long)
    
    def predict(self, image_tensor, text_string):
        """
        Prediction method required by submission system
        
        Args:
            image_tensor: torch.Tensor of shape [3, 224, 224]
            text_string: str, text description
            
        Returns:
            float: probability between 0.0 and 1.0
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            image = image_tensor.unsqueeze(0)  # [1, 3, 224, 224]
            
            # Tokenize text
            text_tokens = self.tokenize(text_string).unsqueeze(0)  # [1, max_length]
            
            # Move to same device as model
            device = next(self.parameters()).device
            image = image.to(device)
            text_tokens = text_tokens.to(device)
            
            # Forward pass
            output = self.forward(image, text_tokens)
            
            # Return as float
            return output.item()
    
    def set_vocab(self, vocab):
        """Set vocabulary for tokenization"""
        self.vocab = vocab


def get_transform():
    """
    Optional transform function for submission system
    Returns the image transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
