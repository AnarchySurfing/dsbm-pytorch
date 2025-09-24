# SpectralEvaluator Guide

## Overview

The `SpectralEvaluator` is a comprehensive evaluation module designed specifically for visible-to-infrared image translation tasks. It provides both standard image quality metrics and specialized cross-spectral evaluation capabilities.

## Features

### Standard Image Quality Metrics
- **FID (Fréchet Inception Distance)**: Measures the quality and diversity of generated images
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Perceptual similarity metric
- **PSNR (Peak Signal-to-Noise Ratio)**: Pixel-level quality measurement
- **SSIM (Structural Similarity Index)**: Structural similarity assessment

### Spectral-Specific Metrics
- **Spectral Consistency**: Measures how well structural information is preserved across spectral domains
  - Gradient consistency using Sobel operators
  - Structural consistency using local binary patterns
- **Cross-Spectral Correlation**: Evaluates statistical relationships between visible and infrared domains
  - Frequency domain correlation using FFT
  - Mutual information between spectral domains

### Visualization Capabilities
- Side-by-side comparison plots (visible, target infrared, generated infrared)
- Difference maps showing pixel-wise and gradient differences
- Metrics history visualization for training monitoring
- Customizable plot generation with various sample counts

## Quick Start

### Basic Usage

```python
from bridge.data.spectral_evaluator import SpectralEvaluator
import torch

# Create evaluator
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluator = SpectralEvaluator(device=device)

# Prepare your data (images should be in [-1, 1] range)
visible_images = torch.randn(4, 3, 256, 256)      # Input visible images
target_infrared = torch.randn(4, 3, 256, 256)     # Ground truth infrared
generated_infrared = torch.randn(4, 3, 256, 256)  # Model-generated infrared

# Perform evaluation
results = evaluator.evaluate_batch(
    visible_images, target_infrared, generated_infrared,
    save_dir="evaluation_results",
    create_visualizations=True
)

# Access results
print(f"FID Score: {results.fid_score:.4f}")
print(f"PSNR: {results.psnr_score:.2f} dB")
print(f"SSIM: {results.ssim_score:.4f}")
print(f"Spectral Consistency: {results.spectral_consistency:.4f}")
```

### Using the Factory Function

```python
from bridge.data.spectral_evaluator import create_spectral_evaluator

# Create evaluator with dataset-specific FID statistics
evaluator = create_spectral_evaluator(
    device='cuda',
    dataset_name='your_dataset',
    data_dir='/path/to/data'
)
```

## Detailed API Reference

### SpectralEvaluator Class

#### Constructor
```python
SpectralEvaluator(device='cuda', fid_features_path=None)
```

**Parameters:**
- `device` (str): Device to run computations on ('cuda' or 'cpu')
- `fid_features_path` (str, optional): Path to precomputed FID features

#### Key Methods

##### evaluate_batch()
```python
evaluate_batch(visible, target_ir, generated_ir, save_dir=None, create_visualizations=True)
```

Performs comprehensive evaluation on a batch of images.

**Parameters:**
- `visible` (torch.Tensor): Input visible images [B, C, H, W]
- `target_ir` (torch.Tensor): Target infrared images [B, C, H, W]
- `generated_ir` (torch.Tensor): Generated infrared images [B, C, H, W]
- `save_dir` (str, optional): Directory to save visualizations
- `create_visualizations` (bool): Whether to create comparison plots

**Returns:**
- `SpectralEvaluationResults`: Container with all metrics and visualizations

##### compute_cross_spectral_metrics()
```python
compute_cross_spectral_metrics(generated_ir, target_ir, visible_input)
```

Computes all evaluation metrics.

**Returns:**
- `dict`: Dictionary containing all computed metrics

##### generate_comparison_plots()
```python
generate_comparison_plots(visible, target_ir, generated_ir, save_dir, filename_prefix="spectral_comparison", num_samples=8)
```

Generates side-by-side comparison visualizations.

**Returns:**
- `list`: List of file paths to saved plots

##### generate_metrics_visualization()
```python
generate_metrics_visualization(metrics_history, save_dir, filename="metrics_history.png")
```

Creates plots showing metrics evolution over time.

**Parameters:**
- `metrics_history` (dict): Dictionary mapping metric names to lists of values

### SpectralEvaluationResults Class

Container for evaluation results with the following attributes:

- `fid_score` (float): FID score
- `lpips_score` (float): LPIPS score
- `psnr_score` (float): PSNR score in dB
- `ssim_score` (float): SSIM score
- `spectral_consistency` (float): Spectral consistency metric
- `cross_spectral_correlation` (float): Cross-spectral correlation metric
- `visual_samples` (list): Sample images for visualization
- `comparison_plots` (list): Paths to generated comparison plots

## Advanced Usage

### Batch Processing During Training

```python
evaluator = SpectralEvaluator(device='cuda')
metrics_history = {
    'fid': [], 'lpips': [], 'psnr': [], 'ssim': [],
    'spectral_consistency': [], 'cross_spectral_correlation': []
}

for epoch in range(num_epochs):
    # ... training code ...
    
    # Evaluate on validation set
    with torch.no_grad():
        results = evaluator.evaluate_batch(
            val_visible, val_target_ir, model_output,
            create_visualizations=(epoch % 10 == 0)  # Save plots every 10 epochs
        )
    
    # Store metrics
    for metric_name in metrics_history.keys():
        metrics_history[metric_name].append(getattr(results, f"{metric_name.replace('_', '')}_score"))
    
    # Generate progress visualization
    if epoch % 50 == 0:
        evaluator.generate_metrics_visualization(
            metrics_history, f"training_progress/epoch_{epoch}"
        )
```

### Individual Metric Components

```python
evaluator = SpectralEvaluator(device='cuda')

# Use individual metric components
gradient_consistency = evaluator.spectral_consistency.compute_gradient_consistency(
    visible_images, generated_infrared
)

frequency_correlation = evaluator.cross_spectral_correlation.compute_frequency_domain_correlation(
    visible_images, generated_infrared
)

mutual_info = evaluator.cross_spectral_correlation.compute_mutual_information(
    visible_images, generated_infrared, bins=64
)
```

### Custom Visualization

```python
# Generate custom comparison plots
comparison_plots = evaluator.generate_comparison_plots(
    visible_batch[:4],      # First 4 samples
    target_ir_batch[:4],
    generated_ir_batch[:4],
    save_dir="custom_results",
    filename_prefix="model_v2_comparison",
    num_samples=4
)

print(f"Saved plots: {comparison_plots}")
```

## Integration with Training Pipeline

### With PyTorch Lightning

```python
import pytorch_lightning as pl
from bridge.data.spectral_evaluator import SpectralEvaluator

class SpectralTranslationModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.evaluator = SpectralEvaluator(device=self.device)
        # ... model initialization ...
    
    def validation_step(self, batch, batch_idx):
        visible, target_ir = batch
        generated_ir = self.forward(visible)
        
        # Compute loss
        loss = self.criterion(generated_ir, target_ir)
        
        # Evaluate every N batches
        if batch_idx % 10 == 0:
            results = self.evaluator.evaluate_batch(
                visible, target_ir, generated_ir,
                create_visualizations=False
            )
            
            # Log metrics
            self.log('val_psnr', results.psnr_score)
            self.log('val_ssim', results.ssim_score)
            self.log('val_spectral_consistency', results.spectral_consistency)
        
        return loss
```

### With Standard PyTorch Training Loop

```python
def evaluate_model(model, val_loader, evaluator, device, save_dir):
    model.eval()
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (visible, target_ir) in enumerate(val_loader):
            visible = visible.to(device)
            target_ir = target_ir.to(device)
            
            generated_ir = model(visible)
            
            results = evaluator.evaluate_batch(
                visible, target_ir, generated_ir,
                save_dir=save_dir if batch_idx == 0 else None,  # Save plots for first batch only
                create_visualizations=(batch_idx == 0)
            )
            
            all_results.append(results)
    
    # Aggregate results
    avg_metrics = {
        'fid': np.mean([r.fid_score for r in all_results]),
        'psnr': np.mean([r.psnr_score for r in all_results]),
        'ssim': np.mean([r.ssim_score for r in all_results]),
        'spectral_consistency': np.mean([r.spectral_consistency for r in all_results]),
    }
    
    return avg_metrics
```

## Performance Considerations

### Memory Usage
- The evaluator processes images in batches to manage GPU memory
- For large images (>512x512), consider reducing batch size
- FID computation requires storing features, which can be memory-intensive

### Computation Time
- LPIPS is the most computationally expensive metric
- Spectral consistency metrics involve convolution operations
- FFT operations in frequency domain correlation scale with image size

### Optimization Tips
1. **Precompute FID Statistics**: Save FID features for your dataset to avoid recomputation
2. **Batch Size**: Use appropriate batch sizes based on available GPU memory
3. **Selective Evaluation**: Don't compute all metrics for every batch during training
4. **Visualization Frequency**: Generate plots periodically, not every evaluation

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU for evaluation if necessary
   - Process images in smaller chunks

2. **Import Errors**
   - Ensure all dependencies are installed: `torch`, `torchmetrics`, `matplotlib`, `numpy`
   - Check that the bridge module is in your Python path

3. **Metric Values Seem Wrong**
   - Verify input images are in [-1, 1] range
   - Check that visible and infrared images are properly paired
   - Ensure target and generated images have the same dimensions

4. **Visualization Issues**
   - Make sure the save directory exists and is writable
   - Check that matplotlib backend supports image saving
   - Verify sufficient disk space for plot files

### Debug Mode

```python
# Enable debug information
import logging
logging.basicConfig(level=logging.DEBUG)

evaluator = SpectralEvaluator(device='cpu')  # Use CPU for debugging
results = evaluator.evaluate_batch(visible, target_ir, generated_ir)
```

## Examples

See `examples/spectral_evaluator_example.py` for comprehensive usage examples including:
- Basic evaluation workflow
- Batch processing simulation
- Factory function usage
- Individual metric component testing
- Visualization generation

## Requirements

- PyTorch >= 1.9.0
- torchmetrics >= 0.9.0
- matplotlib >= 3.3.0
- numpy >= 1.19.0
- seaborn (optional, for enhanced visualizations)

## Citation

If you use the SpectralEvaluator in your research, please cite:

```bibtex
@misc{spectral_evaluator,
  title={SpectralEvaluator: Comprehensive Evaluation for Visible-to-Infrared Image Translation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo/spectral-evaluator}}
}
```