"""
Quick validation of TensorBoard image logging implementation
"""
import tempfile
import shutil
import numpy as np
import torch
from PIL import Image
from bridge.runners.logger import TensorBoardLogger

def validate_implementation():
    """Validate that all required functionality is implemented"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        logger = TensorBoardLogger(save_dir=temp_dir, name="validation")
        
        # Test 1: Basic functionality exists
        assert hasattr(logger, 'log_image'), "log_image method missing"
        assert hasattr(logger, '_convert_image_to_tensor'), "_convert_image_to_tensor method missing"
        
        # Test 2: PIL Image support
        pil_img = Image.new('RGB', (32, 32), color='red')
        logger.log_image("test_pil", [pil_img], step=1)
        print("✓ PIL Image logging works")
        
        # Test 3: Numpy array support
        numpy_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        logger.log_image("test_numpy", [numpy_img], step=1)
        print("✓ Numpy array logging works")
        
        # Test 4: Torch tensor support
        tensor_img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        logger.log_image("test_tensor", [tensor_img], step=1)
        print("✓ Torch tensor logging works")
        
        # Test 5: Multiple images with captions
        images = [pil_img, numpy_img]
        captions = ["PIL", "Numpy"]
        logger.log_image("test_multi", images, step=1, caption=captions)
        print("✓ Multiple images with captions works")
        
        # Test 6: FB parameter support
        logger.log_image("test_fb", [pil_img], step=1, fb="forward")
        print("✓ FB parameter support works")
        
        # Test 7: Error handling for invalid input
        try:
            logger.log_image("test_error", "not_a_list", step=1)
            assert False, "Should have raised TypeError"
        except TypeError:
            print("✓ Error handling for invalid input works")
        
        # Test 8: Error handling for mismatched kwargs
        try:
            logger.log_image("test_mismatch", [pil_img, numpy_img], step=1, caption=["only_one"])
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Error handling for mismatched kwargs works")
        
        print("\n🎉 All validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    validate_implementation()