#!/bin/bash

# Production Training Script for Spectral Image Translation
# Usage: ./train_production.sh [stage] [additional_args...]

# 设置CUDA架构
export TORCH_CUDA_ARCH_LIST="8.9"

export CUDA_VISIBLE_DEVICES=1


# 指定Python路径
PYTHON_PATH="/home/myx123/.conda/envs/brige/bin/python"

# 获取训练阶段参数
STAGE=${1:-"production"}  # 默认使用production阶段
shift  # 移除第一个参数，剩余参数传递给训练

case $STAGE in
    "stable")
        echo "🔧 Running STABLE training (for debugging)..."
        $PYTHON_PATH main.py dataset=visible_infrared_stable model=spectral_unet_stable \
            data.image_size=64 batch_size=32 num_iter=50000 \
            hydra.job.chdir=false "$@"
        ;;
    "production")
        echo "🚀 Running PRODUCTION training (recommended)..."
        $PYTHON_PATH main.py dataset=visible_infrared_production model=spectral_unet_production \
            data.image_size=64 batch_size=32 num_iter=5000 \
            hydra.job.chdir=false "$@"
        ;;
    "transfer")
        echo "🔄 Running TRANSFER training..."
        $PYTHON_PATH main.py dataset=visible_infrared_transfer model=spectral_unet_production \
            data.image_size=64 batch_size=32 num_iter=5000 \
            hydra.job.chdir=false "$@"
        ;;
    *)
        echo "❌ Unknown stage: $STAGE"
        echo "Available stages: stable, production, transfer"
        echo "Usage: ./train_production.sh [stage] [additional_args...]"
        echo ""
        echo "Examples:"
        echo "  ./train_production.sh stable          # Quick test"
        echo "  ./train_production.sh production      # Recommended"
        echo "  ./train_production.sh transfer        # Transfer learning"
        echo "  ./train_production.sh production lr=0.00005  # Custom params"
        exit 1
        ;;
esac