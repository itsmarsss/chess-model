#!/bin/bash
# Quick test script to verify Modal setup works

echo "🧪 Running Modal test - this should complete in ~1 minute"
echo ""
echo "This will:"
echo "  ✓ Download dataset from Hugging Face"
echo "  ✓ Verify C++ compilation"
echo "  ✓ Run 1 epoch with minimal data"
echo "  ✓ Verify training pipeline works"
echo ""

cd "$(dirname "$0")"

# Run test mode
modal run modal_train.py --test-mode

echo ""
echo "✅ Test complete! If you see 'Training completed successfully', the setup works."
echo "   You can now run full training without --test-mode"

