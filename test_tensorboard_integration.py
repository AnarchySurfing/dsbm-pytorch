"""
Integration test for TensorBoard image logging with trainer-like usage
"""
import tempfile
import shutil
import numpy as np
import torch
from PIL import Image
from bridge.runners.logger import TensorBoardLogger


def test_integration_with_trainer_usage():
    """Test TensorBoard image logging in a trainer-like scenario"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize logger like in trainer
        logger = TensorBoardLogger(save_dir=temp_dir, name="integration_test")
        
        # Simulate trainer logging scenarios
        print("Testing basic image logging...")
        
        # Test 1: Single image with step and fb
        img = Image.new('RGB', (64, 64), color='red')
        logger.log_image("sample_image", [img], step=1, fb="forward")
        print("✓ Single image with fb parameter")
        
        # Test 2: Multiple images with captions (like plot results)
        images = [
            np.random.rand(64, 64, 3),  # Generated sample
            np.random.rand(64, 64, 3),  # Real sample
            np.random.rand(64, 64, 3)   # Reconstructed sample
        ]
        captions = ["Generated", "Real", "Reconstructed"]
        logger.log_image("comparison", images, step=2, caption=captions, fb="test")
        print("✓ Multiple images with captions")
        
        # Test 3: Grayscale images (common in scientific applications)
        grayscale_img = np.random.rand(64, 64)
        logger.log_image("grayscale_result", [grayscale_img], step=3)
        print("✓ Grayscale image logging")
        
        # Test 4: Torch tensors (model outputs)
        tensor_img = torch.rand(3, 64, 64)
        logger.log_image("model_output", [tensor_img], step=4, fb="backward")
        print("✓ Torch tensor logging")
        
        # Test 5: Mixed format batch (realistic scenario)
        mixed_images = [
            Image.new('RGB', (32, 32), color='blue'),  # PIL
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),  # Numpy
            torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)  # Tensor
        ]
        mixed_captions = ["PIL Input", "Numpy Processing", "Tensor Output"]
        logger.log_image("pipeline_results", mixed_images, step=5, caption=mixed_captions)
        print("✓ Mixed format batch logging")
        
        print("\n🎉 All integration tests passed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_integration_with_trainer_usage()