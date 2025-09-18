#!/bin/bash

# 4-Scenario LTXV Debugging Script Runner
# This script sets up the environment and runs the debugging script

echo "🚀 Starting 4-Scenario LTXV Debug Session"
echo "=========================================="

# Check and activate conda environment
if command -v conda &> /dev/null; then
    # Check if ltxv environment exists
    if conda env list | grep -q "ltxv"; then
        echo "🐍 Activating conda environment: ltxv"
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ltxv
    else
        echo "⚠️  ltxv conda environment not found. Using current environment."
    fi
else
    echo "⚠️  conda not found. Using current environment."
fi

# Check if we're in the correct directory
if [ ! -f "debug_four_scenarios.py" ]; then
    echo "❌ Error: debug_four_scenarios.py not found. Please run from the project root directory."
    exit 1
fi

# Set PYTHONPATH to include src directory
export PYTHONPATH=./src:$PYTHONPATH

# Check for LoRA checkpoints
LORA_PATH=""
if [ -d "outputs/checkpoints" ]; then
    # Specify the exact LoRA file you want to use
    DESIRED_LORA="lora_weights_step_00300.safetensors"  # Change this line to use different LoRA

    if [ -f "outputs/checkpoints/$DESIRED_LORA" ]; then
        LORA_FILE="outputs/checkpoints/$DESIRED_LORA"
        LORA_PATH="--lora-path $LORA_FILE"
        echo "📦 Using specified LoRA checkpoint: $LORA_FILE"
    else
        # Fallback to first available LoRA
        LORA_FILE=$(find outputs/checkpoints -name "*lora*.safetensors" | head -1)
        if [ -n "$LORA_FILE" ]; then
            LORA_PATH="--lora-path $LORA_FILE"
            echo "📦 Fallback to available LoRA checkpoint: $LORA_FILE"
        else
            echo "⚠️  No LoRA checkpoint found in outputs/checkpoints/"
        fi
    fi
else
    echo "⚠️  outputs/checkpoints/ directory not found"
fi

# Default prompt (can be overridden by command line argument)
DEFAULT_PROMPT="'a blonde boy talking to his friends in a living room"
PROMPT=${1:-"$DEFAULT_PROMPT"}

echo "📝 Using prompt: $PROMPT"
echo "🔧 Using LoRA: ${LORA_FILE:-"None"}"
echo ""

# Create output directory
mkdir -p debug_outputs

# Run the debug script
echo "🎬 Starting video generation for all 4 scenarios..."

python3 debug_four_scenarios.py \
    --prompt "$PROMPT" \
    --base-model "Lightricks/LTX-Video-0.9.5" \
    $LORA_PATH \
    --device cuda \
    --output-dir debug_outputs \
    --dtype bfloat16

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Debug session completed successfully!"
    echo "📁 Check the debug_outputs/ directory for generated videos and reports."
    echo ""
    echo "📊 Results Summary:"
    ls -la debug_outputs/*.mp4 2>/dev/null || echo "No videos generated"
    ls -la debug_outputs/*.json 2>/dev/null || echo "No report generated"
else
    echo ""
    echo "❌ Debug session failed. Check the error messages above."
    exit 1
fi