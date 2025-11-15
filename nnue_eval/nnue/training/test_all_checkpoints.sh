#!/bin/bash
# Test all available checkpoints to find the best one
# Usage: ./test_all_checkpoints.sh

TRAINING_DIR="/Users/itsmarsss/Documents/chesshacks/nnue_eval/nnue/training"
INFERENCE_DIR="/Users/itsmarsss/Documents/chesshacks/nnue_eval/nnue/inference/my-chesshacks-bot"
MODELS_DIR="/Users/itsmarsss/Documents/chesshacks/nnue_eval/nnue/inference/models"

cd "$TRAINING_DIR"

echo "🔍 Finding all checkpoint files..."
echo ""

# Find all .ckpt files
CKPT_FILES=$(find . -name "*.ckpt" -type f | sort)

if [ -z "$CKPT_FILES" ]; then
    echo "❌ No checkpoint files found!"
    echo "Make sure you have .ckpt files in the training directory"
    exit 1
fi

echo "Found checkpoints:"
echo "$CKPT_FILES"
echo ""

echo "💡 Serializing with epoch numbers (e.g., last.e5.nnue for epoch 5)..."
echo ""

# Serialize each checkpoint with epoch info
for ckpt in $CKPT_FILES; do
    basename=$(basename "$ckpt" .ckpt)
    nnue_base="$MODELS_DIR/${basename}"
    
    echo "📦 Serializing $ckpt with epoch info..."
    python serialize.py "$ckpt" "${nnue_base}.nnue" --export-epochs
    
    if [ $? -eq 0 ]; then
        # Find the generated file (it will have .eN.nnue suffix)
        generated=$(ls -t "$MODELS_DIR/${basename}".e*.nnue 2>/dev/null | head -1)
        if [ -n "$generated" ]; then
            echo "✅ Created $generated"
            
            # Get file size
            size=$(ls -lh "$generated" | awk '{print $5}')
            echo "   Size: $size"
            
            # Quick architecture check
            python check_nnue_arch.py "$generated" 2>/dev/null | head -3
        else
            echo "✅ Created ${nnue_base}.nnue (no epoch info in checkpoint)"
        fi
    else
        echo "❌ Failed to serialize $ckpt"
    fi
    echo ""
done

echo ""
echo "✅ All checkpoints serialized!"
echo ""
echo "📊 Now test each model manually:"
echo ""
echo "Available models in $MODELS_DIR:"
ls -1 "$MODELS_DIR"/*.e*.nnue 2>/dev/null | while read model; do
    basename=$(basename "$model")
    echo "  python serve.py --model $basename --port 5058"
done
echo ""
echo "💡 Tip: The BEST model is often NOT the last checkpoint!"
echo "   Filenames with .e6, .e7, .e8, .e9 (epochs 6-9) often perform better"
echo "   than .e12, .e13, .e14, .e15 (later epochs) due to overfitting."
