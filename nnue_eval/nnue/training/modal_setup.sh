#!/bin/bash
# Setup script for Modal deployment

echo "Setting up Modal for NNUE training..."

# Install Modal CLI if not already installed
if ! command -v modal &> /dev/null; then
    echo "Installing Modal CLI..."
    pip install modal
fi

# Authenticate with Modal (if not already done)
echo "Make sure you're authenticated with Modal:"
echo "  modal token set"

# Create Hugging Face secret (if using private datasets)
echo ""
echo "To use private Hugging Face datasets, create a secret:"
echo "  modal secret create huggingface HF_TOKEN=your_token_here"

echo ""
echo "Setup complete!"
echo ""
echo "To run training:"
echo "  modal run nnue_eval/nnue/training/modal_train.py --dataset-name linrock/test80-2024"

