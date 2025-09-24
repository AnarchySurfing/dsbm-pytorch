#!/bin/bash

# Quick verification script to test experiment path configuration

echo "🔍 Verifying experiment path configuration..."

# Set environment
export TORCH_CUDA_ARCH_LIST="8.9"
PYTHON_PATH="/home/myx123/.conda/envs/brige/bin/python"

# Run a very short training to test paths
echo "Running 1-iteration test..."
$PYTHON_PATH main.py dataset=visible_infrared_stable model=spectral_unet_stable \
    data.image_size=64 batch_size=2 num_iter=1 \
    hydra.job.chdir=false

# Check if files are in the right place
echo ""
echo "📁 Checking experiment directory structure..."

# Find the experiment directory
EXPERIMENT_DIR=$(find experiments -name "*visible_infrared*" -type d | head -1)

if [ -n "$EXPERIMENT_DIR" ]; then
    echo "✓ Found experiment directory: $EXPERIMENT_DIR"
    
    # Check for expected subdirectories
    for subdir in cache checkpoints im gif tensorboard_logs; do
        if [ -d "$EXPERIMENT_DIR/$subdir" ]; then
            echo "✓ Found $subdir directory"
        else
            echo "⚠ Missing $subdir directory (may be created later)"
        fi
    done
    
    # Check for any files in project root that should be in experiment dir
    echo ""
    echo "🔍 Checking for misplaced files in project root..."
    
    MISPLACED_FILES=0
    for pattern in "cache_*.png" "*.gif" "checkpoints" "cache" "im" "gif"; do
        if ls $pattern 2>/dev/null | grep -q .; then
            echo "⚠ Found misplaced files: $pattern"
            MISPLACED_FILES=1
        fi
    done
    
    if [ $MISPLACED_FILES -eq 0 ]; then
        echo "✓ No misplaced files found in project root"
    fi
    
else
    echo "✗ No experiment directory found!"
    echo "Checking if files were created in project root..."
    ls -la | grep -E "(cache|checkpoints|im|gif)"
fi

echo ""
echo "📋 Summary:"
echo "- Experiment files should be in: experiments/{experiment_name}/{parameters}/{seed}/"
echo "- Project root should remain clean"
echo "- Logs and outputs should be in the experiment directory"