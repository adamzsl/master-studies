"""
Unit tests for model components
"""
import torch
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import SubmissionModel, ImageEncoder, TextEncoder, FusionModule, get_transform


class TestModelComponents(unittest.TestCase):
    """Test individual model components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.image_size = 224
        self.seq_length = 50
        self.vocab_size = 1000
        
    def test_image_encoder(self):
        """Test ImageEncoder forward pass"""
        encoder = ImageEncoder(embedding_dim=256, pretrained=False)
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        
        output = encoder(x)
        
        self.assertEqual(output.shape, (self.batch_size, 256))
        self.assertFalse(torch.isnan(output).any())
        
    def test_text_encoder(self):
        """Test TextEncoder forward pass"""
        encoder = TextEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            bidirectional=True
        )
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        output = encoder(x)
        
        self.assertEqual(output.shape, (self.batch_size, 128))
        self.assertFalse(torch.isnan(output).any())
        
    def test_fusion_module(self):
        """Test FusionModule forward pass"""
        fusion = FusionModule(embedding_dim=256, hidden_dim=256)
        img_emb = torch.randn(self.batch_size, 256)
        txt_emb = torch.randn(self.batch_size, 256)
        
        output = fusion(img_emb, txt_emb)
        
        self.assertEqual(output.shape, (self.batch_size,))
        self.assertTrue((output >= 0).all() and (output <= 1).all())
        
    def test_full_model(self):
        """Test full SubmissionModel"""
        model = SubmissionModel(
            vocab_size=self.vocab_size,
            image_embedding_dim=256,
            text_embedding_dim=128
        )
        
        images = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        texts = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        output = model(images, texts)
        
        self.assertEqual(output.shape, (self.batch_size,))
        self.assertTrue((output >= 0).all() and (output <= 1).all())
        self.assertFalse(torch.isnan(output).any())
        
    def test_predict_method(self):
        """Test predict method for submission"""
        model = SubmissionModel(vocab_size=self.vocab_size)
        model.eval()
        
        # Create dummy vocab
        vocab = {f"word{i}": i for i in range(100)}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        model.set_vocab(vocab)
        
        # Test single prediction
        image = torch.randn(3, self.image_size, self.image_size)
        text = "word1 word2 word3"
        
        score = model.predict(image, text)
        
        self.assertIsInstance(score, float)
        self.assertTrue(0.0 <= score <= 1.0)
        
    def test_tokenization(self):
        """Test tokenization method"""
        model = SubmissionModel(vocab_size=self.vocab_size)
        
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            'hello': 2,
            'world': 3,
            'test': 4
        }
        model.set_vocab(vocab)
        
        # Test basic tokenization
        tokens = model.tokenize("hello world")
        self.assertEqual(tokens.shape, (model.max_length,))
        self.assertEqual(tokens[0].item(), 2)  # hello
        self.assertEqual(tokens[1].item(), 3)  # world
        
        # Test unknown word
        tokens = model.tokenize("unknown")
        self.assertEqual(tokens[0].item(), 1)  # <UNK>
        
        # Test padding
        tokens = model.tokenize("hello")
        self.assertEqual(tokens[1].item(), 0)  # <PAD>
        
    def test_get_transform(self):
        """Test transform function"""
        transform = get_transform()
        
        # Create dummy image
        from PIL import Image
        img = Image.new('RGB', (300, 400), color='red')
        
        transformed = transform(img)
        
        self.assertEqual(transformed.shape, (3, 224, 224))
        self.assertTrue(transformed.min() >= -3)  # After normalization
        self.assertTrue(transformed.max() <= 3)


class TestSubmissionFormat(unittest.TestCase):
    """Test submission format requirements"""
    
    def test_submission_model_exists(self):
        """Test that SubmissionModel class exists"""
        from model import SubmissionModel
        self.assertTrue(callable(SubmissionModel))
        
    def test_predict_method_exists(self):
        """Test that predict method exists"""
        model = SubmissionModel()
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(callable(model.predict))
        
    def test_get_transform_exists(self):
        """Test that get_transform function exists"""
        from model import get_transform
        self.assertTrue(callable(get_transform))
        
    def test_model_is_nn_module(self):
        """Test that model inherits from nn.Module"""
        import torch.nn as nn
        model = SubmissionModel()
        self.assertIsInstance(model, nn.Module)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
