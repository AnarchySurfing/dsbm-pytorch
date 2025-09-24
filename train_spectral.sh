#!/bin/bash

# 设置CUDA架构以消除警告 (RTX 40系列)
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_VISIBLE_DEVICES=1
# 指定Python路径
PYTHON_PATH="/home/myx123/.conda/envs/brige/bin/python"

# 运行光谱图像翻译训练 (稳定配置)
$PYTHON_PATH main.py dataset=visible_infrared_stable model=spectral_unet_stable \
  data.image_size=128 \
  batch_size=4 \
  num_iter=1000 \
  "$@"  # 允许传递额外参数