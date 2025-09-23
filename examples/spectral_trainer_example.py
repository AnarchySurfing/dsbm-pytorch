"""
Example usage of the SpectralDBDSBTrainer for visible-infrared image translation.

This script demonstrates how to set up and use the SpectralDBDSBTrainer
for training Schrödinger Bridge models on paired visible-infrared datasets.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from types import SimpleNamespace
import tempfile
from PIL import Image

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge.data.spectral import VisibleInfraredDataset
from bridge.data.preprocessing import SpectralImagePreprocessor, SpectralAugmentationPipeline
from bridge.trainer_spectral import SpectralDBDSBTrainer


def create_example_dataset(root_dir: Path, num_pairs: int = 20, image_size: tuple = (128, 128)):
    """
    Create an example visible-infrared dataset for demonstration.
    
    In practice, you would replace this with your actual dataset loading.
    """
    print(f"Creating example dataset with {num_pairs} pairs...")
    
    visible_dir = root_dir / "visible"
    infrared_dir = root_dir / "infrared"
    
    visible_dir.mkdir(parents=True, exist_ok=True)
    infrared_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_pairs):
        # Create synthetic visible image (colorful)
        visible_data = np.random.randint(50, 255, (*image_size, 3), dtype=np.uint8)
        
        # Add some structure (simple patterns)
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 < (min(image_size) // 4) ** 2
        visible_data[mask] = visible_data[mask] * 0.7 + 50
        
        visible_img = Image.fromarray(visible_data, 'RGB')
        visible_img.save(visible_dir / f"pair_{i:04d}.png")
        
        # Create corresponding infrared image (correlated but different)
        # Simulate thermal characteristics: warmer objects appear brighter in IR
        infrared_base = visible_data.mean(axis=2) * 0.6 + np.random.randint(0, 80, image_size)
        infrared_base = np.clip(infrared_base, 0, 255).astype(np.uint8)
        
        # Make center region "warmer" (brighter in IR)
        infrared_base[mask] = np.clip(infrared_base[mask] + 50, 0, 255)
        
        infrared_data = np.stack([infrared_base] * 3, axis=-1)
        infrared_img = Image.fromarray(infrared_data, 'RGB')
        infrared_img.save(infrared_dir / f"pair_{i:04d}.png")
    
    print(f"✓ Created dataset at {root_dir}")


def create_training_config():
    """Create a training configuration for the spectral trainer."""
    
    class TrainingConfig:
        def __init__(self):
            pass
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = TrainingConfig()
    
    # Basic training parameters
    config.n_ipf = 3  # Number of IPF iterations
    config.num_steps = 20  # Number of diffusion steps
    config.batch_size = 4  # Batch size
    config.num_repeat_data = 1
    config.num_iter = 10  # Iterations per IPF step
    config.first_num_iter = 10
    config.normalize_x1 = False
    
    # SDE parameters
    config.sde = "ve"  # Variance Exploding SDE
    config.gamma_min = 0.01
    config.gamma_max = 1.0
    config.gamma_space = "linspace"
    config.symmetric_gamma = False
    
    # Loss and optimization
    config.loss_scale = False
    config.grad_clipping = True
    config.grad_clip = 1.0
    
    # Cache parameters
    config.cache_npar = None
    config.cache_refresh_stride = 5
    config.cache_batch_size = 8
    config.cache_num_steps = 20
    
    # Test parameters
    config.test_npar = 16
    config.test_batch_size = 8
    config.test_num_steps = 20
    
    # Visualization
    config.plot_npar = 8
    config.gif_stride = 5
    config.log_stride = 2
    
    # Model architecture (simplified for example)
    config.model = "unet"
    config.num_channels = 3
    config.num_res_blocks = 2
    config.num_heads = 4
    config.num_head_channels = 32
    config.attention_resolutions = "16,8"
    config.dropout = 0.1
    config.channel_mult = "1,2,2"
    config.conv_resample = True
    config.dims = 2
    config.use_checkpoint = False
    config.use_scale_shift_norm = True
    config.resblock_updown = False
    config.use_fp16 = False
    config.use_new_attention_order = False
    
    # Optimizer
    config.optimizer = "adam"
    config.lr = 1e-4
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.weight_decay = 0.0
    
    # EMA
    config.ema = True
    config.ema_rate = 0.9999
    
    # Other parameters
    config.cdsb = False  # Not using conditional DSB
    config.transfer = False
    config.first_coupling = "ref"
    config.mean_match = False
    config.use_prev_net = False
    config.autostart_next_it = True
    config.std_trick = False
    
    # Data loading
    config.num_workers = 2
    config.pin_memory = True
    
    # Spectral-specific parameters
    config.spectral_loss_weight = 1.0
    config.perceptual_loss_weight = 0.1
    config.consistency_loss_weight = 0.05
    config.use_spectral_augmentation = True
    
    return config


def example_basic_setup():
    """Demonstrate basic setup of SpectralDBDSBTrainer."""
    print("=== Basic SpectralDBDSBTrainer Setup ===")
    
    # Create example dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = Path(temp_dir)
        create_example_dataset(dataset_path, num_pairs=20)
        
        # Load dataset
        dataset = VisibleInfraredDataset(
            root_dir=str(dataset_path),
            image_size=128,
            validate_pairs=True
        )
        
        print(f"Loaded dataset with {len(dataset)} pairs")
        
        # Create preprocessor
        preprocessor = SpectralImagePreprocessor(
            target_size=128,
            normalize_range=(-1.0, 1.0),
            interpolation_mode='bilinear'
        )
        
        # Create augmentation pipeline
        augmenter = SpectralAugmentationPipeline(
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.2,
            rotation_degrees=10,
            brightness_factor=0.1,
            contrast_factor=0.1,
            saturation_factor=0.1,
            hue_factor=0.05,
            apply_color_jitter_to_infrared=False
        )
        
        print("✓ Created preprocessor and augmentation pipeline")
        
        # Test preprocessing
        visible, infrared = dataset[0]
        vis_proc, ir_proc = preprocessor.preprocess_pair(visible, infrared)
        vis_aug, ir_aug = augmenter(vis_proc, ir_proc)
        
        print(f"✓ Preprocessing test:")
        print(f"  Original: {visible.shape} -> Processed: {vis_proc.shape}")
        print(f"  Range: [{vis_proc.min():.3f}, {vis_proc.max():.3f}]")
        
        # Note: Full trainer initialization requires proper accelerator setup
        print("✓ Basic setup completed successfully")


def example_loss_computation():
    """Demonstrate spectral loss computation."""
    print("\n=== Spectral Loss Computation Example ===")
    
    # Create mock data
    batch_size = 4
    channels = 3
    height = width = 128
    
    visible = torch.randn(batch_size, channels, height, width) * 0.5
    infrared_gt = torch.randn(batch_size, channels, height, width) * 0.5
    infrared_pred = infrared_gt + torch.randn_like(infrared_gt) * 0.2
    t = torch.rand(batch_size)
    
    # Create mock trainer for loss computation
    class MockTrainer:
        def __init__(self):
            self.spectral_loss_weight = 1.0
            self.perceptual_loss_weight = 0.1
            self.consistency_loss_weight = 0.05
            self.perceptual_net = None  # Disable for example
            self.logger = None
            
            # Mock args
            class MockArgs:
                loss_scale = False
            self.args = MockArgs()
    
    trainer = MockTrainer()
    
    # Import the trainer class
    from bridge.trainer_spectral import SpectralDBDSBTrainer
    
    # Compute spectral loss
    std = torch.ones(batch_size, 1, 1, 1)
    total_loss = SpectralDBDSBTrainer.compute_loss(
        trainer, infrared_pred, infrared_gt, visible, t, 'f', std
    )
    
    print(f"✓ Computed spectral loss: {total_loss.item():.4f}")
    
    # Compute individual loss components
    losses = SpectralDBDSBTrainer.compute_spectral_loss(
        trainer, infrared_pred, infrared_gt, visible, t
    )
    
    print("✓ Individual loss components:")
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")


def example_metrics_evaluation():
    """Demonstrate spectral metrics evaluation."""
    print("\n=== Spectral Metrics Evaluation Example ===")
    
    # Create mock data with different quality levels
    batch_size = 8
    channels = 3
    height = width = 128
    
    visible = torch.randn(batch_size, channels, height, width) * 0.5
    infrared_gt = torch.randn(batch_size, channels, height, width) * 0.5
    
    # High quality generation (low noise)
    infrared_good = infrared_gt + torch.randn_like(infrared_gt) * 0.05
    
    # Low quality generation (high noise)
    infrared_poor = infrared_gt + torch.randn_like(infrared_gt) * 0.3
    
    from bridge.trainer_spectral import SpectralDBDSBTrainer
    
    # Evaluate metrics for good generation
    metrics_good = SpectralDBDSBTrainer.evaluate_spectral_metrics(
        None, visible, infrared_gt, infrared_good
    )
    
    # Evaluate metrics for poor generation
    metrics_poor = SpectralDBDSBTrainer.evaluate_spectral_metrics(
        None, visible, infrared_gt, infrared_poor
    )
    
    print("✓ Metrics comparison:")
    print("  High Quality Generation:")
    for metric, value in metrics_good.items():
        print(f"    {metric}: {value:.4f}")
    
    print("  Low Quality Generation:")
    for metric, value in metrics_poor.items():
        print(f"    {metric}: {value:.4f}")
    
    # Verify that good generation has better metrics
    assert metrics_good['mse'] < metrics_poor['mse'], "Good generation should have lower MSE"
    assert metrics_good['psnr'] > metrics_poor['psnr'], "Good generation should have higher PSNR"
    print("✓ Metrics correctly reflect generation quality")


def example_inference_simulation():
    """Simulate inference process."""
    print("\n=== Inference Simulation Example ===")
    
    # Create mock visible images
    batch_size = 4
    channels = 3
    height = width = 128
    
    visible_images = torch.randn(batch_size, channels, height, width) * 0.5
    
    print(f"Input visible images shape: {visible_images.shape}")
    print(f"Input range: [{visible_images.min():.3f}, {visible_images.max():.3f}]")
    
    # Simulate infrared generation (in practice, this would use the trained model)
    # For demonstration, we'll create plausible infrared images
    infrared_generated = visible_images * 0.7 + torch.randn_like(visible_images) * 0.1
    
    print(f"Generated infrared shape: {infrared_generated.shape}")
    print(f"Generated range: [{infrared_generated.min():.3f}, {infrared_generated.max():.3f}]")
    
    # Simulate postprocessing for visualization
    from bridge.data.preprocessing import SpectralImagePreprocessor
    
    preprocessor = SpectralImagePreprocessor(normalize_range=(-1.0, 1.0))
    vis_uint8, ir_uint8 = preprocessor.postprocess_pair(visible_images, infrared_generated)
    
    print(f"✓ Postprocessed for visualization:")
    print(f"  Visible uint8 range: [{vis_uint8.min()}, {vis_uint8.max()}]")
    print(f"  Infrared uint8 range: [{ir_uint8.min()}, {ir_uint8.max()}]")
    
    # Simulate saving (would save actual images in practice)
    print("✓ Ready for saving/visualization")


def example_training_workflow():
    """Demonstrate the complete training workflow setup."""
    print("\n=== Training Workflow Setup Example ===")
    
    # This example shows how you would set up training in practice
    # Note: Actual training requires proper distributed setup with accelerate
    
    print("1. Dataset Preparation:")
    print("   - Organize visible images in: dataset/visible/")
    print("   - Organize infrared images in: dataset/infrared/")
    print("   - Ensure matching filenames for pairs")
    
    print("\n2. Configuration:")
    config = create_training_config()
    print(f"   - IPF iterations: {config.n_ipf}")
    print(f"   - Diffusion steps: {config.num_steps}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Learning rate: {config.lr}")
    
    print("\n3. Preprocessing Setup:")
    preprocessor = SpectralImagePreprocessor(
        target_size=128,
        normalize_range=(-1.0, 1.0)
    )
    print("   ✓ Spectral preprocessor configured")
    
    print("\n4. Augmentation Setup:")
    augmenter = SpectralAugmentationPipeline(
        horizontal_flip_prob=0.5,
        brightness_factor=0.1,
        contrast_factor=0.1
    )
    print("   ✓ Augmentation pipeline configured")
    
    print("\n5. Training Process:")
    print("   - Initialize SpectralDBDSBTrainer with datasets and config")
    print("   - Run trainer.train() for full IPF training loop")
    print("   - Monitor losses: MSE, perceptual, consistency")
    print("   - Save checkpoints and generated samples")
    
    print("\n6. Inference:")
    print("   - Load trained model checkpoint")
    print("   - Use trainer.generate_infrared() for visible->IR translation")
    print("   - Evaluate with spectral metrics")
    
    print("\n✓ Complete workflow setup demonstrated")


if __name__ == "__main__":
    print("SpectralDBDSBTrainer Usage Examples")
    print("=" * 60)
    
    try:
        example_basic_setup()
        example_loss_computation()
        example_metrics_evaluation()
        example_inference_simulation()
        example_training_workflow()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("\nKey takeaways:")
        print("1. SpectralDBDSBTrainer extends IPF_DBDSB for spectral translation")
        print("2. Supports spectral-aware loss computation with multiple components")
        print("3. Provides comprehensive metrics for evaluation")
        print("4. Integrates with existing DSBM infrastructure")
        print("5. Ready for visible-to-infrared image translation tasks")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {str(e)}")
        import traceback
        traceback.print_exc()