"""Simple, server-safe submission model.

This file must work when imported in an isolated submission folder.
Required by evaluator:
  - get_transform()
  - class SubmissionModel(nn.Module) with predict(image_tensor, text_string) -> float
"""

import re
import hashlib

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class SubmissionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 10000,
        text_embed_dim: int = 256,
        text_hidden_dim: int = 256,
        max_length: int = 50,
        cnn_backbone: str = 'resnet50',
        pretrained: bool = False,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)

        # Image encoder (avoid downloads on the server: default pretrained=False)
        backbone = (cnn_backbone or 'resnet50').lower()
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        else:
            resnet = models.resnet50(pretrained=pretrained)

        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        image_feat_dim = resnet.fc.in_features
        self.image_proj = nn.Linear(image_feat_dim, text_embed_dim)

        # Text encoder (FIXED vocab_size=10000 by default)
        self.text_embedding = nn.Embedding(self.vocab_size, text_embed_dim, padding_idx=0)
        self.text_lstm = nn.LSTM(
            input_size=text_embed_dim,
            hidden_size=text_hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )
        self.text_proj = nn.Linear(text_hidden_dim * 2, text_embed_dim)

        # Fusion + classifier (logits)
        fusion_in = text_embed_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

        # Optional training-time vocab (if you want explicit IDs)
        self.vocab = None

    def set_vocab(self, vocab: dict):
        self.vocab = vocab

    def _hash_token(self, token: str) -> int:
        # Stable hash across runs (unlike Python's built-in hash())
        h = hashlib.md5(token.encode('utf-8')).digest()
        # Reserve: 0=PAD, 1=UNK
        return 2 + (int.from_bytes(h[:4], 'little') % max(1, self.vocab_size - 2))

    def tokenize(self, text: str) -> torch.Tensor:
        # Keep it simple and robust
        words = re.sub(r'[^a-zA-Z\s]', ' ', (text or '').lower()).split()
        ids = []
        for w in words[: self.max_length]:
            if self.vocab is not None:
                idx = int(self.vocab.get(w, 1))
                if idx >= self.vocab_size:
                    idx = 1
                ids.append(idx)
            else:
                ids.append(self._hash_token(w))

        if len(ids) < self.max_length:
            ids.extend([0] * (self.max_length - len(ids)))

        return torch.tensor(ids, dtype=torch.long)

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, 224, 224]
        # text_tokens: [B, L]
        img = self.image_encoder(images).flatten(1)
        img = self.image_proj(img)

        emb = self.text_embedding(text_tokens)
        _, (h, _) = self.text_lstm(emb)
        # last layer, biLSTM: concat forward/backward
        h = torch.cat([h[-2], h[-1]], dim=1)
        txt = self.text_proj(h)

        mult = img * txt
        fused = torch.cat([img, txt, mult], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits

    def predict(self, image_tensor: torch.Tensor, text_string: str) -> float:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            img = image_tensor.unsqueeze(0).to(device)
            tok = self.tokenize(text_string).unsqueeze(0).to(device)
            logits = self.forward(img, tok)
            return torch.sigmoid(logits).item()
