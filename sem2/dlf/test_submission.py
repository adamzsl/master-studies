"""
Testing and evaluation utilities
"""
import torch
from model import SubmissionModel, get_transform
from PIL import Image


def test_model_loading(model_path='weights.pth', vocab_path='vocab.json'):
    """Test if model can be loaded correctly"""
    import json
    
    print("Testing model loading...")
    
    # Load vocab
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Create model
    model = SubmissionModel(vocab_size=len(vocab))
    model.set_vocab(vocab)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("✓ Model loaded successfully!")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def test_prediction(model, image_path, text):
    """Test prediction on a single example"""
    transform = get_transform()
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    
    # Predict
    score = model.predict(image_tensor, text)
    
    print(f"\nImage: {image_path}")
    print(f"Text: {text}")
    print(f"Match Score: {score:.4f}")
    print(f"Prediction: {'MATCH' if score > 0.5 else 'NO MATCH'}")
    
    return score


def evaluate_submission(model_path='model.py', weights_path='weights.pth'):
    """
    Simulate submission system evaluation
    Similar to submission_evaluator.py from reference repo
    """
    import importlib.util
    import sys
    
    print("Evaluating submission format...")
    
    # Check files exist
    import os
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(weights_path):
        print(f"✗ Weights file not found: {weights_path}")
        return False
    
    print("✓ Required files found")
    
    # Load model module
    spec = importlib.util.spec_from_file_location("submission_model", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_model"] = module
    spec.loader.exec_module(module)
    
    # Check SubmissionModel class exists
    if not hasattr(module, 'SubmissionModel'):
        print("✗ SubmissionModel class not found in model.py")
        return False
    
    print("✓ SubmissionModel class found")
    
    # Check get_transform function (optional)
    if hasattr(module, 'get_transform'):
        print("✓ get_transform function found")
        transform = module.get_transform()
    else:
        print("  Note: get_transform not defined, will use default")
        transform = get_transform()
    
    # Instantiate model
    try:
        model = module.SubmissionModel()
        print("✓ Model instantiated")
    except Exception as e:
        print(f"✗ Failed to instantiate model: {e}")
        return False
    
    # Load weights
    try:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        print("✓ Weights loaded")
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        return False
    
    # Check predict method
    if not hasattr(model, 'predict'):
        print("✗ predict method not found")
        return False
    
    print("✓ predict method found")
    
    # Test predict method with dummy data
    try:
        dummy_image = torch.randn(3, 224, 224)
        dummy_text = "a test caption"
        result = model.predict(dummy_image, dummy_text)
        
        if not isinstance(result, float):
            print(f"✗ predict should return float, got {type(result)}")
            return False
        
        if not (0.0 <= result <= 1.0):
            print(f"✗ predict should return value in [0, 1], got {result}")
            return False
        
        print(f"✓ predict method works correctly (returned {result:.4f})")
    except Exception as e:
        print(f"✗ predict method failed: {e}")
        return False
    
    print("\n✓✓✓ All checks passed! Submission format is valid.")
    return True


def create_submission_zip(output_name='submission.zip'):
    """Create submission ZIP file"""
    import zipfile
    import os
    
    files_to_include = ['model.py', 'weights.pth']
    optional_files = ['vocab.json', 'config.py']
    
    print(f"Creating submission ZIP: {output_name}")
    
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Required files
        for filename in files_to_include:
            if os.path.exists(filename):
                zipf.write(filename)
                print(f"  Added: {filename}")
            else:
                print(f"  ✗ Missing: {filename}")
                return False
        
        # Optional files
        for filename in optional_files:
            if os.path.exists(filename):
                zipf.write(filename)
                print(f"  Added: {filename} (optional)")
    
    print(f"\n✓ Submission ZIP created: {output_name}")
    print(f"  Size: {os.path.getsize(output_name) / 1024 / 1024:.2f} MB")
    
    return True


if __name__ == '__main__':
    # Test model loading
    try:
        model = test_model_loading()
    except Exception as e:
        print(f"Model loading test failed: {e}")
        print("\nMake sure you have:")
        print("  - weights.pth (trained model weights)")
        print("  - vocab.json (vocabulary)")
    
    # Evaluate submission format
    print("\n" + "="*50)
    evaluate_submission()
