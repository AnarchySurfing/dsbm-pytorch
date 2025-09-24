# Spectral Configuration System Usage Guide

## Overview

The spectral configuration system provides seamless integration of visible-to-infrared image translation capabilities with the existing DSBM framework. It includes dataset configurations, model configurations, and complete pipeline support.

## Quick Start

### Basic Usage

```bash
# Train a spectral translation model
python main.py dataset=visible_infrared model=spectral_unet_small

# Transfer learning experiment
python main.py dataset=visible_infrared_transfer model=spectral_unet_small

# High-resolution training
python main.py dataset=visible_infrared model=spectral_unet data.image_size=512 batch_size=1
```

### Configuration Files

#### Dataset Configurations

1. **`visible_infrared.yaml`** - Basic visible-to-infrared translation
   - Standard training setup
   - Full resolution support
   - Spectral augmentations

2. **`visible_infrared_transfer.yaml`** - Transfer learning setup
   - Smaller batch sizes
   - Lower learning rates
   - Progressive unfreezing support

#### Model Configurations

1. **`spectral_unet.yaml`** - Full spectral U-Net
   - 128 base channels
   - Multi-scale attention
   - Spectral-specific features

2. **`spectral_unet_small.yaml`** - Lightweight version
   - 64 base channels
   - Faster training
   - Reduced memory usage

## Dataset Setup

### Directory Structure

```
data/
└── visible_infrared/
    ├── visible/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    └── infrared/
        ├── image001.jpg
        ├── image002.jpg
        └── ...
```

### Requirements

- Images with matching filenames are treated as pairs
- Supported formats: JPG, PNG, TIFF
- Recommended resolution: 256x256 or higher
- RGB format for both visible and infrared (infrared converted to 3-channel)

## Configuration Parameters

### Dataset Parameters

```yaml
data:
  image_size: 256              # Target image size
  channels: 3                  # RGB channels
  cond_channels: 3             # Conditioning channels
  random_flip: true            # Data augmentation
  spectral_augmentation: true  # Spectral-specific augmentations
  normalize_method: "standard" # Normalization method
  
  # Dataset paths
  root_dir: "data/visible_infrared"
  visible_subdir: "visible"
  infrared_subdir: "infrared"
```

### Model Parameters

```yaml
model:
  num_channels: 128            # Base model channels
  channel_mult: [1, 2, 2, 4, 4]  # Channel multipliers
  num_res_blocks: 3            # Residual blocks per level
  attention_resolutions: "32,16,8"  # Attention resolutions
  dropout: 0.1                 # Dropout rate
  use_checkpoint: false        # Gradient checkpointing
```

### Training Parameters

```yaml
batch_size: 16                 # Training batch size
num_iter: 100000              # Training iterations
lr: 0.0001                    # Learning rate
num_steps: 100                # Sampling steps
n_ipf: 50                     # IPF iterations
```

## Advanced Usage

### Custom Configurations

Create custom configuration files by extending existing ones:

```yaml
# @package _global_
# Custom high-resolution config

defaults:
  - visible_infrared
  - override model: spectral_unet

# Override specific parameters
data:
  image_size: 512
  
model:
  num_channels: 192
  use_checkpoint: true
  
batch_size: 2
memory:
  mixed_precision: true
  gradient_accumulation_steps: 4
```

### Command Line Overrides

```bash
# Override any parameter from command line
python main.py dataset=visible_infrared model=spectral_unet \
  data.image_size=512 \
  batch_size=2 \
  lr=0.00005 \
  model.num_channels=192

# Enable specific features
python main.py dataset=visible_infrared model=spectral_unet \
  data.spectral_augmentation=true \
  model.use_checkpoint=true \
  memory.mixed_precision=true
```

### Multi-GPU Training

```bash
# Distributed training
python -m torch.distributed.launch --nproc_per_node=4 main.py \
  dataset=visible_infrared \
  model=spectral_unet \
  batch_size=8
```

## Evaluation

The spectral configuration system automatically integrates with the spectral evaluation module:

```yaml
evaluation:
  compute_spectral_metrics: true
  save_comparison_plots: true
  metrics_history_frequency: 100
  evaluation_frequency: 1000
  
  metrics:
    - "fid"
    - "lpips" 
    - "psnr"
    - "ssim"
    - "spectral_consistency"
    - "cross_spectral_correlation"
```

## Memory Optimization

For large images or limited GPU memory:

```yaml
memory:
  gradient_accumulation_steps: 4
  mixed_precision: true
  max_memory_per_gpu: 0.9
  enable_memory_efficient_attention: true

model:
  use_checkpoint: true
  
batch_size: 1  # Reduce batch size
```

## Logging and Monitoring

```yaml
# TensorBoard logging
LOGGER: TensorBoard
tensorboard_log_dir: ./spectral_logs
tensorboard_name: spectral_experiment

# Logging frequency
log_stride: 50
gif_stride: 1000
evaluation_frequency: 1000
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size`
   - Enable `model.use_checkpoint=true`
   - Enable `memory.mixed_precision=true`
   - Reduce `data.image_size`

2. **Dataset Not Found**
   - Check `data.root_dir` path
   - Verify directory structure
   - Ensure image files have matching names

3. **Slow Training**
   - Use `model=spectral_unet_small`
   - Reduce `data.image_size`
   - Increase `batch_size` if memory allows
   - Enable `memory.mixed_precision=true`

### Performance Tips

1. **Fast Experimentation**
   ```bash
   python main.py dataset=visible_infrared model=spectral_unet_small \
     data.image_size=64 batch_size=16 num_iter=10000
   ```

2. **High Quality Training**
   ```bash
   python main.py dataset=visible_infrared model=spectral_unet \
     data.image_size=512 batch_size=2 num_iter=200000 \
     model.use_checkpoint=true memory.mixed_precision=true
   ```

3. **Transfer Learning**
   ```bash
   python main.py dataset=visible_infrared_transfer model=spectral_unet_small \
     lr=0.00005 num_iter=50000
   ```

## Integration with Existing Code

The spectral configuration system is fully compatible with existing DSBM code:

- Uses existing model architectures (UNet)
- Integrates with existing training loops
- Compatible with existing logging systems
- Works with existing checkpoint/resume functionality

## Examples

See `examples/spectral_config_example.py` for comprehensive usage examples and `examples/spectral_evaluator_example.py` for evaluation examples.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example scripts
3. Verify your dataset structure and configuration files