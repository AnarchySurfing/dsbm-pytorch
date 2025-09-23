"""
Example usage of the image preprocessing module.

This script demonstrates how to use the SpectralImagePreprocessor and
SpectralAugmentationPipeline for preparing visible-infrared image pairs
for training and inference.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge.data.spectral import VisibleInfraredDataset
from bridge.data.preprocessing import (
    SpectralImagePreprocessor,
    SpectralAugmentationPipeline,
    normalize_rgb_to_range,
    denormalize_to_uint8,
    compute_image_statistics
)


def example_basic_preprocessing():
    """Demonstrate basic preprocessing operations."""
    print("=== Basic Preprocessing Example ===")
    
    # Create synthetic test data
    visible = torch.randint(0, 256, (3, 128, 128), dtype=torch.float32)
    infrared = torch.randint(50, 200, (128, 128), dtype=torch.float32)
    infrared = torch.stack([infrared] * 3, dim=0)  # Convert to 3-channel
    
    print(f"Original shapes: visible {visible.shape}, infrared {infrared.shape}")
    print(f"Original ranges: visible [{visible.min():.1f}, {visible.max():.1f}], "
          f"infrared [{infrared.min():.1f}, {infrared.max():.1f}]")
    
    # Initialize preprocessor
    preprocessor = SpectralImagePreprocessor(
        target_size=256,  # Resize to 256x256
        normalize_range=(-1.0, 1.0),  # Normalize to [-1, 1] for training
        interpolation_mode='bilinear'
    )
    
    # Preprocess the pair
    vis_processed, ir_processed = preprocessor.preprocess_pair(visible, infrared)
    
    print(f"Processed shapes: visible {vis_processed.shape}, infrared {ir_processed.shape}")
    print(f"Processed ranges: visible [{vis_processed.min():.3f}, {vis_processed.max():.3f}], "
          f"infrared [{ir_processed.min():.3f}, {ir_processed.max():.3f}]")
    
    # Postprocess back to uint8 for saving/visualization
    vis_uint8, ir_uint8 = preprocessor.postprocess_pair(vis_processed, ir_processed)
    
    print(f"Postprocessed types: visible {vis_uint8.dtype}, infrared {ir_uint8.dtype}")
    print(f"Postprocessed ranges: visible [{vis_uint8.min()}, {vis_uint8.max()}], "
          f"infrared [{ir_uint8.min()}, {ir_uint8.max()}]")


def example_augmentation_pipeline():
    """Demonstrate augmentation pipeline usage."""
    print("\n=== Augmentation Pipeline Example ===")
    
    # Create test data in [-1, 1] range (as would come from preprocessor)
    visible = torch.rand(3, 128, 128) * 2 - 1
    infrared = torch.rand(3, 128, 128) * 2 - 1
    
    # Initialize augmentation pipeline
    augmenter = SpectralAugmentationPipeline(
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.2,
        rotation_degrees=15,  # ±15 degrees rotation
        brightness_factor=0.1,
        contrast_factor=0.1,
        saturation_factor=0.1,
        hue_factor=0.05,
        apply_color_jitter_to_infrared=False  # Usually don't apply color jitter to IR
    )
    
    print("Applying augmentations...")
    
    # Apply augmentations multiple times to see variation
    for i in range(3):
        vis_aug, ir_aug = augmenter(visible, infrared)
        print(f"Augmentation {i+1}: visible range [{vis_aug.min():.3f}, {vis_aug.max():.3f}], "
              f"infrared range [{ir_aug.min():.3f}, {ir_aug.max():.3f}]")


def example_dataset_integration():
    """Demonstrate integration with dataset and DataLoader."""
    print("\n=== Dataset Integration Example ===")
    
    # This example assumes you have a dataset at the specified path
    dataset_path = "/path/to/your/visible_infrared_dataset"
    
    try:
        # Create dataset with preprocessing
        dataset = VisibleInfraredDataset(
            root_dir=dataset_path,
            image_size=256,
            validate_pairs=True
        )
        
        print(f"Dataset loaded with {len(dataset)} pairs")
        
        # Compute dataset statistics
        print("Computing dataset statistics...")
        stats = compute_image_statistics(dataset, num_samples=min(100, len(dataset)))
        
        print(f"Visible - Mean: {stats['visible']['mean']}")
        print(f"Visible - Std: {stats['visible']['std']}")
        print(f"Infrared - Mean: {stats['infrared']['mean']}")
        print(f"Infrared - Std: {stats['infrared']['std']}")
        
        # Create preprocessor and augmenter
        preprocessor = SpectralImagePreprocessor(
            target_size=256,
            normalize_range=(-1.0, 1.0)
        )
        
        augmenter = SpectralAugmentationPipeline(
            horizontal_flip_prob=0.5,
            brightness_factor=0.1,
            contrast_factor=0.1
        )
        
        # Custom dataset class that applies preprocessing and augmentation
        class PreprocessedDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, preprocessor, augmenter=None):
                self.base_dataset = base_dataset
                self.preprocessor = preprocessor
                self.augmenter = augmenter
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                visible, infrared = self.base_dataset[idx]
                
                # Apply preprocessing
                visible, infrared = self.preprocessor.preprocess_pair(visible, infrared)
                
                # Apply augmentation if provided
                if self.augmenter is not None:
                    visible, infrared = self.augmenter(visible, infrared)
                
                return visible, infrared
        
        # Create preprocessed dataset
        preprocessed_dataset = PreprocessedDataset(dataset, preprocessor, augmenter)
        
        # Create DataLoader
        dataloader = DataLoader(
            preprocessed_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )
        
        print("Testing DataLoader with preprocessing...")
        for batch_idx, (vis_batch, ir_batch) in enumerate(dataloader):
            print(f"Batch {batch_idx}: visible {vis_batch.shape}, infrared {ir_batch.shape}")
            print(f"  Visible range: [{vis_batch.min():.3f}, {vis_batch.max():.3f}]")
            print(f"  Infrared range: [{ir_batch.min():.3f}, {ir_batch.max():.3f}]")
            
            if batch_idx >= 2:  # Only show first 3 batches
                break
        
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")
        print("Please update the dataset_path variable with your actual dataset location")


def example_training_pipeline():
    """Demonstrate a complete training preprocessing pipeline."""
    print("\n=== Training Pipeline Example ===")
    
    # Training configuration
    config = {
        'image_size': 256,
        'batch_size': 8,
        'normalize_range': (-1.0, 1.0),
        'augmentation': {
            'horizontal_flip_prob': 0.5,
            'rotation_degrees': 10,
            'brightness_factor': 0.1,
            'contrast_factor': 0.1,
        }
    }
    
    print(f"Training configuration: {config}")
    
    # Create preprocessing components
    preprocessor = SpectralImagePreprocessor(
        target_size=config['image_size'],
        normalize_range=config['normalize_range']
    )
    
    augmenter = SpectralAugmentationPipeline(
        horizontal_flip_prob=config['augmentation']['horizontal_flip_prob'],
        rotation_degrees=config['augmentation']['rotation_degrees'],
        brightness_factor=config['augmentation']['brightness_factor'],
        contrast_factor=config['augmentation']['contrast_factor']
    )
    
    print("✓ Preprocessing pipeline configured for training")
    
    # Simulate training data processing
    print("Simulating training batch processing...")
    
    for epoch in range(2):  # Simulate 2 epochs
        print(f"\nEpoch {epoch + 1}:")
        
        for batch_idx in range(3):  # Simulate 3 batches per epoch
            # Simulate loading raw data (would come from DataLoader)
            batch_visible = torch.randint(0, 256, (config['batch_size'], 3, 128, 128), dtype=torch.float32)
            batch_infrared = torch.randint(50, 200, (config['batch_size'], 3, 128, 128), dtype=torch.float32)
            
            # Process each sample in the batch
            processed_visible = []
            processed_infrared = []
            
            for i in range(config['batch_size']):
                # Preprocess
                vis, ir = preprocessor.preprocess_pair(batch_visible[i], batch_infrared[i])
                
                # Augment (only during training)
                vis, ir = augmenter(vis, ir)
                
                processed_visible.append(vis)
                processed_infrared.append(ir)
            
            # Stack into batch tensors
            vis_batch = torch.stack(processed_visible)
            ir_batch = torch.stack(processed_infrared)
            
            print(f"  Batch {batch_idx + 1}: {vis_batch.shape} -> model training")
            
            # Here you would pass vis_batch and ir_batch to your model
            # loss = model(vis_batch, ir_batch)
            # loss.backward()
            # optimizer.step()


def example_inference_pipeline():
    """Demonstrate inference preprocessing pipeline."""
    print("\n=== Inference Pipeline Example ===")
    
    # Create test input (simulating loaded image)
    visible_input = torch.randint(0, 256, (3, 512, 512), dtype=torch.float32)
    
    print(f"Input image shape: {visible_input.shape}")
    print(f"Input range: [{visible_input.min():.1f}, {visible_input.max():.1f}]")
    
    # Inference preprocessing (no augmentation)
    preprocessor = SpectralImagePreprocessor(
        target_size=256,  # Model input size
        normalize_range=(-1.0, 1.0)
    )
    
    # For inference, we only have visible image, infrared is what we want to generate
    visible_processed = preprocessor._normalize_tensor(visible_input)
    if preprocessor.target_size is not None:
        visible_processed = torch.nn.functional.interpolate(
            visible_processed.unsqueeze(0),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    print(f"Preprocessed for inference: {visible_processed.shape}")
    print(f"Preprocessed range: [{visible_processed.min():.3f}, {visible_processed.max():.3f}]")
    
    # Simulate model inference
    # infrared_generated = model(visible_processed.unsqueeze(0))
    # For demo, create fake generated infrared
    infrared_generated = torch.randn_like(visible_processed)
    
    print(f"Generated infrared shape: {infrared_generated.shape}")
    
    # Postprocess for visualization/saving
    vis_output, ir_output = preprocessor.postprocess_pair(visible_processed, infrared_generated)
    
    print(f"Output shapes: visible {vis_output.shape}, infrared {ir_output.shape}")
    print(f"Output types: visible {vis_output.dtype}, infrared {ir_output.dtype}")
    print(f"Ready for saving/visualization")


if __name__ == "__main__":
    print("Image Preprocessing Module Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_preprocessing()
    example_augmentation_pipeline()
    example_dataset_integration()
    example_training_pipeline()
    example_inference_pipeline()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nKey takeaways:")
    print("1. Use SpectralImagePreprocessor for consistent preprocessing of image pairs")
    print("2. Use SpectralAugmentationPipeline for training-time augmentations")
    print("3. Integrate with VisibleInfraredDataset for complete data pipeline")
    print("4. Apply different preprocessing for training vs inference")
    print("5. Always postprocess generated images before visualization/saving")