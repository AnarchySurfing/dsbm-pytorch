"""
Example usage of the paired spectral dataset infrastructure.

This script demonstrates how to use the VisibleInfraredDataset class
for loading paired visible-infrared image datasets.
"""

import torch
from torch.utils.data import DataLoader
from bridge.data.spectral import VisibleInfraredDataset
import matplotlib.pyplot as plt
import numpy as np


def denormalize_tensor(tensor):
    """Convert tensor from [-1, 1] range back to [0, 1] for visualization."""
    return (tensor + 1.0) / 2.0


def visualize_pairs(dataset, num_pairs=3):
    """
    Visualize visible-infrared pairs from the dataset.
    
    Args:
        dataset: VisibleInfraredDataset instance
        num_pairs: Number of pairs to visualize
    """
    fig, axes = plt.subplots(2, num_pairs, figsize=(15, 6))
    
    for i in range(min(num_pairs, len(dataset))):
        visible, infrared = dataset[i]
        metadata = dataset.get_metadata(i)
        
        # Convert tensors to numpy arrays for visualization
        visible_np = denormalize_tensor(visible).permute(1, 2, 0).numpy()
        infrared_np = denormalize_tensor(infrared).permute(1, 2, 0).numpy()
        
        # Plot visible image
        axes[0, i].imshow(visible_np)
        axes[0, i].set_title(f'Visible: {metadata["visible_filename"]}')
        axes[0, i].axis('off')
        
        # Plot infrared image
        axes[1, i].imshow(infrared_np)
        axes[1, i].set_title(f'Infrared: {metadata["infrared_filename"]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('spectral_pairs_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def example_basic_usage():
    """Demonstrate basic usage of the VisibleInfraredDataset."""
    print("=== Basic Usage Example ===")
    
    # Initialize dataset
    # Replace with your actual dataset path
    dataset_path = "/path/to/your/visible_infrared_dataset"
    
    try:
        dataset = VisibleInfraredDataset(
            root_dir=dataset_path,
            image_size=256,  # Resize images to 256x256
            validate_pairs=True  # Validate all pairs during initialization
        )
        
        print(f"Dataset loaded successfully with {len(dataset)} pairs")
        
        # Get a single sample
        visible, infrared = dataset[0]
        print(f"Sample shapes: visible {visible.shape}, infrared {infrared.shape}")
        print(f"Value ranges: visible [{visible.min():.3f}, {visible.max():.3f}], "
              f"infrared [{infrared.min():.3f}, {infrared.max():.3f}]")
        
        # Get metadata
        metadata = dataset.get_metadata(0)
        print(f"Sample metadata: {metadata}")
        
        # Visualize some pairs
        visualize_pairs(dataset, num_pairs=3)
        
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")
        print("Please update the dataset_path variable with your actual dataset location")


def example_dataloader_usage():
    """Demonstrate usage with PyTorch DataLoader."""
    print("\n=== DataLoader Usage Example ===")
    
    dataset_path = "/path/to/your/visible_infrared_dataset"
    
    try:
        dataset = VisibleInfraredDataset(
            root_dir=dataset_path,
            image_size=128,
            validate_pairs=True
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )
        
        print(f"DataLoader created with batch_size=4")
        
        # Iterate through one batch
        for batch_idx, (visible_batch, infrared_batch) in enumerate(dataloader):
            print(f"Batch {batch_idx}: visible {visible_batch.shape}, infrared {infrared_batch.shape}")
            
            # Process batch here...
            # For example, you could pass it to your model:
            # output = model(visible_batch, infrared_batch)
            
            if batch_idx >= 2:  # Only show first 3 batches
                break
                
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")


def example_custom_transforms():
    """Demonstrate usage with custom transforms."""
    print("\n=== Custom Transforms Example ===")
    
    import torchvision.transforms as transforms
    
    # Define custom transforms
    visible_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
    ])
    
    infrared_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # Note: Be careful with color transforms on infrared images
    ])
    
    dataset_path = "/path/to/your/visible_infrared_dataset"
    
    try:
        dataset = VisibleInfraredDataset(
            root_dir=dataset_path,
            image_size=256,
            transform=visible_transform,
            target_transform=infrared_transform,
            validate_pairs=True
        )
        
        print(f"Dataset with transforms loaded: {len(dataset)} pairs")
        
        # Get samples to see transform effects
        visible1, infrared1 = dataset[0]
        visible2, infrared2 = dataset[0]  # Same index, different transforms
        
        print("Transforms applied - samples will be different each time due to randomness")
        
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")


def example_dataset_statistics():
    """Demonstrate how to compute dataset statistics."""
    print("\n=== Dataset Statistics Example ===")
    
    dataset_path = "/path/to/your/visible_infrared_dataset"
    
    try:
        dataset = VisibleInfraredDataset(
            root_dir=dataset_path,
            validate_pairs=True
        )
        
        print(f"Computing statistics for {len(dataset)} pairs...")
        
        # Compute mean and std for visible and infrared channels
        visible_sum = torch.zeros(3)
        infrared_sum = torch.zeros(3)
        visible_sq_sum = torch.zeros(3)
        infrared_sq_sum = torch.zeros(3)
        
        for i in range(min(100, len(dataset))):  # Sample first 100 images
            visible, infrared = dataset[i]
            
            visible_sum += visible.mean(dim=[1, 2])
            infrared_sum += infrared.mean(dim=[1, 2])
            visible_sq_sum += (visible ** 2).mean(dim=[1, 2])
            infrared_sq_sum += (infrared ** 2).mean(dim=[1, 2])
        
        n_samples = min(100, len(dataset))
        visible_mean = visible_sum / n_samples
        infrared_mean = infrared_sum / n_samples
        visible_std = torch.sqrt(visible_sq_sum / n_samples - visible_mean ** 2)
        infrared_std = torch.sqrt(infrared_sq_sum / n_samples - infrared_mean ** 2)
        
        print(f"Visible - Mean: {visible_mean}, Std: {visible_std}")
        print(f"Infrared - Mean: {infrared_mean}, Std: {infrared_std}")
        
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")


if __name__ == "__main__":
    print("Paired Spectral Dataset Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_dataloader_usage()
    example_custom_transforms()
    example_dataset_statistics()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use with your own dataset:")
    print("1. Organize your data in the following structure:")
    print("   dataset_root/")
    print("   ├── visible/")
    print("   │   ├── image1.jpg")
    print("   │   └── image2.jpg")
    print("   └── infrared/")
    print("       ├── image1.jpg")
    print("       └── image2.jpg")
    print("2. Update the dataset_path variable in the examples")
    print("3. Run the examples to verify your dataset works correctly")