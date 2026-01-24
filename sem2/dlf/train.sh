#!/bin/bash
# Quick start script for training the model

echo "=================================="
echo "Cross-Modal Verification Training"
echo "=================================="

# Check Python
if ! command -v python &> /dev/null; then
    echo "Python not found! Please install Python 3.7+"
    exit 1
fi

echo "✓ Python found"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -q -r requirements.txt

echo "✓ Requirements installed"

# Check if data exists
if [ ! -d "data/images" ]; then
    echo ""
    echo "⚠️  Data directory not found!"
    echo ""
    read -p "Create minimal test dataset? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python prepare_data.py
    else
        echo ""
        echo "Please prepare data manually:"
        echo "  1. Download Flickr8k or COCO dataset"
        echo "  2. Place images in data/images/"
        echo "  3. Create data/captions.json"
        echo "  4. Or run: python prepare_data.py"
        exit 1
    fi
fi

echo ""
echo "=================================="
echo "Starting training..."
echo "=================================="
echo ""

# Run training
python train.py

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Check training_curves.png for training progress"
echo "  2. Test submission: python test_submission.py"
echo "  3. Create ZIP: zip submission.zip model.py weights.pth vocab.json"
echo ""
