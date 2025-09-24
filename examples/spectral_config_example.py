"""
Example script demonstrating the spectral configuration system.

This script shows how to use the new visible-infrared dataset and spectral U-Net
configurations with the existing Hydra configuration system.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from bridge.runners.config_getters import (
    get_model, get_datasets, get_plotter, get_logger
)


def create_sample_dataset(base_dir, num_samples=10):
    """Create a sample visible-infrared dataset for demonstration."""
    visible_dir = os.path.join(base_dir, 'visible_infrared', 'visible')
    infrared_dir = os.path.join(base_dir, 'visible_infrared', 'infrared')
    
    os.makedirs(visible_dir, exist_ok=True)
    os.makedirs(infrared_dir, exist_ok=True)
    
    print(f"Creating {num_samples} sample image pairs...")
    
    # Create sample image pairs
    for i in range(num_samples):
        filename = f'sample_{i:03d}.jpg'
        
        # Create visible image (RGB with some structure)
        visible_img = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
        # Add some structure (vertical stripes)
        visible_img[:, ::8, :] = 255
        
        # Create infrared image (similar structure but different intensity)
        infrared_img = np.random.randint(30, 150, (128, 128, 3), dtype=np.uint8)
        # Add thermal-like patterns
        infrared_img[:, ::8, :] = 200
        infrared_img[::16, :, :] = 100
        
        # Save images
        try:
            from PIL import Image
            Image.fromarray(visible_img).save(os.path.join(visible_dir, filename))
            Image.fromarray(infrared_img).save(os.path.join(infrared_dir, filename))
        except ImportError:
            # Fallback: save as numpy arrays
            np.save(os.path.join(visible_dir, filename.replace('.jpg', '.npy')), visible_img)
            np.save(os.path.join(infrared_dir, filename.replace('.jpg', '.npy')), infrared_img)
    
    print(f"✓ Created sample dataset at {base_dir}")
    return visible_dir, infrared_dir


def demonstrate_basic_config():
    """Demonstrate basic spectral configuration loading."""
    print("\n" + "="*60)
    print("BASIC SPECTRAL CONFIGURATION")
    print("="*60)
    
    GlobalHydra.instance().clear()
    
    try:
        config_dir = str(project_root / "conf")
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Load basic visible-infrared configuration
            cfg = compose(config_name="config", overrides=[
                "dataset=visible_infrared",
                "model=spectral_unet_small",
                "data.image_size=128",
                "batch_size=4",
                "num_iter=1000"
            ])
            
            print("Configuration loaded successfully!")
            print(f"  Dataset: {cfg.Dataset}")
            print(f"  Model: {cfg.Model}")
            print(f"  Image size: {cfg.data.image_size}")
            print(f"  Batch size: {cfg.batch_size}")
            print(f"  Training iterations: {cfg.num_iter}")
            print(f"  Learning rate: {cfg.lr}")
            
            # Show spectral-specific parameters
            if hasattr(cfg.data, 'spectral'):
                print(f"  Spectral channels: {cfg.data.spectral.visible_channels} -> {cfg.data.spectral.infrared_channels}")
                print(f"  Wavelength range: {cfg.data.spectral.wavelength_range}")
            
            return cfg
            
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        raise
    finally:
        GlobalHydra.instance().clear()


def demonstrate_transfer_learning_config():
    """Demonstrate transfer learning configuration."""
    print("\n" + "="*60)
    print("TRANSFER LEARNING CONFIGURATION")
    print("="*60)
    
    GlobalHydra.instance().clear()
    
    try:
        config_dir = str(project_root / "conf")
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Load transfer learning configuration
            cfg = compose(config_name="config", overrides=[
                "dataset=visible_infrared_transfer",
                "model=spectral_unet_small",
                "data.image_size=128",
                "batch_size=2"
            ])
            
            print("Transfer learning configuration loaded!")
            print(f"  Transfer mode: {cfg.transfer}")
            print(f"  Batch size: {cfg.batch_size}")
            print(f"  Learning rate: {cfg.lr}")
            print(f"  Training iterations: {cfg.num_iter}")
            
            # Show transfer learning specific parameters
            if hasattr(cfg, 'transfer_learning'):
                print(f"  Progressive unfreezing: {cfg.transfer_learning.progressive_unfreezing}")
                print(f"  Encoder LR multiplier: {cfg.transfer_learning.encoder_lr_multiplier}")
            
            return cfg
            
    except Exception as e:
        print(f"✗ Transfer learning configuration failed: {e}")
        raise
    finally:
        GlobalHydra.instance().clear()


def demonstrate_model_variants():
    """Demonstrate different model size variants."""
    print("\n" + "="*60)
    print("MODEL VARIANTS DEMONSTRATION")
    print("="*60)
    
    GlobalHydra.instance().clear()
    
    variants = [
        ("Small Model", "spectral_unet_small"),
        ("Standard Model", "spectral_unet"),
    ]
    
    try:
        config_dir = str(project_root / "conf")
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            for name, model_config in variants:
                print(f"\n  {name}:")
                
                try:
                    cfg = compose(config_name="config", overrides=[
                        "dataset=visible_infrared",
                        f"model={model_config}",
                        "data.image_size=64"
                    ])
                    
                    # Create model to check parameters
                    model = get_model(cfg)
                    total_params = sum(p.numel() for p in model.parameters())
                    
                    print(f"    ✓ Model: {cfg.Model}")
                    print(f"    ✓ Channels: {cfg.model.num_channels}")
                    print(f"    ✓ Parameters: {total_params:,}")
                    
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    
    except Exception as e:
        print(f"✗ Model variants demonstration failed: {e}")
        raise
    finally:
        GlobalHydra.instance().clear()


def demonstrate_full_pipeline():
    """Demonstrate the complete pipeline with actual data loading."""
    print("\n" + "="*60)
    print("FULL PIPELINE DEMONSTRATION")
    print("="*60)
    
    GlobalHydra.instance().clear()
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample dataset
            visible_dir, infrared_dir = create_sample_dataset(temp_dir, num_samples=8)
            
            config_dir = str(project_root / "conf")
            
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                # Load configuration
                cfg = compose(config_name="config", overrides=[
                    "dataset=visible_infrared",
                    "model=spectral_unet_small",
                    "data.image_size=64",
                    "batch_size=2",
                    "num_workers=0"  # Avoid multiprocessing issues in demo
                ])
                
                # Update paths
                cfg.data.root_dir = os.path.join(temp_dir, 'visible_infrared')
                cfg.paths.data_dir_name = temp_dir
                
                print("1. Loading dataset...")
                init_ds, final_ds, mean_final, var_final = get_datasets(cfg)
                print(f"   ✓ Dataset loaded: {len(init_ds)} pairs")
                
                print("2. Creating model...")
                model = get_model(cfg)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"   ✓ Model created: {total_params:,} parameters")
                
                print("3. Testing data loading...")
                sample_visible, sample_infrared = init_ds[0]
                print(f"   ✓ Sample shapes: {sample_visible.shape} -> {sample_infrared.shape}")
                
                print("4. Testing model structure...")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                print(f"   ✓ Model moved to device: {device}")
                
                print("5. Creating logger...")
                cfg.tensorboard_log_dir = os.path.join(temp_dir, 'logs')
                logger = get_logger(cfg, "spectral_demo")
                print(f"   ✓ Logger created: {type(logger).__name__}")
                
                print("\n✓ Full pipeline demonstration completed successfully!")
                
    except Exception as e:
        print(f"✗ Full pipeline demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        GlobalHydra.instance().clear()


def demonstrate_config_customization():
    """Demonstrate how to customize configurations."""
    print("\n" + "="*60)
    print("CONFIGURATION CUSTOMIZATION")
    print("="*60)
    
    GlobalHydra.instance().clear()
    
    try:
        config_dir = str(project_root / "conf")
        
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            # Show how to customize various aspects
            customizations = [
                {
                    "name": "High Resolution",
                    "overrides": [
                        "dataset=visible_infrared",
                        "model=spectral_unet",
                        "data.image_size=512",
                        "batch_size=1",
                        "model.num_channels=64"
                    ]
                },
                {
                    "name": "Fast Training",
                    "overrides": [
                        "dataset=visible_infrared",
                        "model=spectral_unet_small",
                        "data.image_size=64",
                        "batch_size=16",
                        "num_iter=5000",
                        "lr=0.001"
                    ]
                },
                {
                    "name": "Memory Efficient",
                    "overrides": [
                        "dataset=visible_infrared",
                        "model=spectral_unet_small",
                        "data.image_size=128",
                        "batch_size=4",
                        "model.use_checkpoint=true"
                    ]
                }
            ]
            
            for custom in customizations:
                print(f"\n  {custom['name']}:")
                
                try:
                    cfg = compose(config_name="config", overrides=custom["overrides"])
                    
                    print(f"    Image size: {cfg.data.image_size}")
                    print(f"    Batch size: {cfg.batch_size}")
                    print(f"    Model channels: {cfg.model.num_channels}")
                    print(f"    Use checkpoint: {cfg.model.get('use_checkpoint', False)}")
                    print(f"    Learning rate: {cfg.lr}")
                    print("    ✓ Configuration valid")
                    
                except Exception as e:
                    print(f"    ✗ Configuration error: {e}")
                    
    except Exception as e:
        print(f"✗ Configuration customization failed: {e}")
        raise
    finally:
        GlobalHydra.instance().clear()


def main():
    """Run all configuration demonstrations."""
    print("Spectral Configuration System Examples")
    print("This script demonstrates the new visible-infrared configuration system.")
    
    try:
        # Run demonstrations
        demonstrate_basic_config()
        demonstrate_transfer_learning_config()
        demonstrate_model_variants()
        demonstrate_full_pipeline()
        demonstrate_config_customization()
        
        print("\n" + "="*60)
        print("✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe spectral configuration system is ready for use!")
        print("\nTo use in your own experiments:")
        print("1. Use 'dataset=visible_infrared' for basic spectral translation")
        print("2. Use 'dataset=visible_infrared_transfer' for transfer learning")
        print("3. Use 'model=spectral_unet_small' for fast experiments")
        print("4. Use 'model=spectral_unet' for full-scale training")
        print("5. Customize parameters via Hydra overrides")
        
    except Exception as e:
        print(f"\n✗ DEMONSTRATIONS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)