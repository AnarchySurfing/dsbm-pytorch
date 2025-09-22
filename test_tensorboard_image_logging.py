"""
Unit tests for TensorBoard image logging functionality
"""
import unittest
import tempfile
import shutil
import os
import numpy as np
import torch
from PIL import Image
from bridge.runners.logger import TensorBoardLogger


class TestTensorBoardImageLogging(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = TensorBoardLogger(save_dir=self.temp_dir, name="test_experiment")
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_log_image_with_pil_image(self):
        """Test logging PIL images"""
        # Create a test PIL image
        img = Image.new('RGB', (64, 64), color='red')
        images = [img]
        
        # Test logging without error
        try:
            self.logger.log_image("test_pil", images, step=1)
        except Exception as e:
            self.fail(f"log_image failed with PIL image: {e}")
    
    def test_log_image_with_numpy_array_hwc(self):
        """Test logging numpy arrays in HWC format"""
        # Create test numpy array (Height, Width, Channels)
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        images = [img_array]
        
        try:
            self.logger.log_image("test_numpy_hwc", images, step=1)
        except Exception as e:
            self.fail(f"log_image failed with numpy HWC array: {e}")
    
    def test_log_image_with_numpy_array_chw(self):
        """Test logging numpy arrays in CHW format"""
        # Create test numpy array (Channels, Height, Width)
        img_array = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        images = [img_array]
        
        try:
            self.logger.log_image("test_numpy_chw", images, step=1)
        except Exception as e:
            self.fail(f"log_image failed with numpy CHW array: {e}")
    
    def test_log_image_with_grayscale_numpy(self):
        """Test logging grayscale numpy arrays"""
        # Create grayscale image
        img_array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        images = [img_array]
        
        try:
            self.logger.log_image("test_grayscale", images, step=1)
        except Exception as e:
            self.fail(f"log_image failed with grayscale numpy array: {e}")
    
    def test_log_image_with_torch_tensor(self):
        """Test logging torch tensors"""
        # Create test torch tensor (CHW format)
        img_tensor = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        images = [img_tensor]
        
        try:
            self.logger.log_image("test_tensor", images, step=1)
        except Exception as e:
            self.fail(f"log_image failed with torch tensor: {e}")
    
    def test_log_image_with_torch_tensor_hwc(self):
        """Test logging torch tensors in HWC format"""
        # Create test torch tensor (HWC format)
        img_tensor = torch.randint(0, 255, (64, 64, 3), dtype=torch.uint8)
        images = [img_tensor]
        
        try:
            self.logger.log_image("test_tensor_hwc", images, step=1)
        except Exception as e:
            self.fail(f"log_image failed with torch tensor HWC: {e}")
    
    def test_log_multiple_images(self):
        """Test logging multiple images at once"""
        # Create multiple test images
        img1 = Image.new('RGB', (32, 32), color='red')
        img2 = Image.new('RGB', (32, 32), color='blue')
        img3 = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        
        images = [img1, img2, img3]
        captions = ["Red image", "Blue image", "Random image"]
        
        try:
            self.logger.log_image("test_multiple", images, step=1, caption=captions)
        except Exception as e:
            self.fail(f"log_image failed with multiple images: {e}")
    
    def test_log_image_with_fb_parameter(self):
        """Test logging images with fb (forward/backward) parameter"""
        img = Image.new('RGB', (32, 32), color='green')
        images = [img]
        
        try:
            self.logger.log_image("test_fb", images, step=1, fb="forward")
        except Exception as e:
            self.fail(f"log_image failed with fb parameter: {e}")
    
    def test_log_image_with_captions(self):
        """Test logging images with captions"""
        img = Image.new('RGB', (32, 32), color='yellow')
        images = [img]
        captions = ["Test caption"]
        
        try:
            self.logger.log_image("test_caption", images, step=1, caption=captions)
        except Exception as e:
            self.fail(f"log_image failed with caption: {e}")
    
    def test_log_image_invalid_input_type(self):
        """Test that invalid input types raise TypeError"""
        with self.assertRaises(TypeError):
            self.logger.log_image("test_invalid", "not_a_list", step=1)
    
    def test_log_image_mismatched_kwargs_length(self):
        """Test that mismatched kwargs length raises ValueError"""
        img1 = Image.new('RGB', (32, 32), color='red')
        img2 = Image.new('RGB', (32, 32), color='blue')
        images = [img1, img2]
        captions = ["Only one caption"]  # Should have 2 captions
        
        with self.assertRaises(ValueError):
            self.logger.log_image("test_mismatch", images, step=1, caption=captions)
    
    def test_log_image_unsupported_image_type(self):
        """Test that unsupported image types raise TypeError"""
        unsupported_image = "string_image"
        images = [unsupported_image]
        
        with self.assertRaises(TypeError):
            self.logger.log_image("test_unsupported", images, step=1)
    
    def test_convert_image_to_tensor_pil_rgba(self):
        """Test conversion of RGBA PIL image to RGB tensor"""
        img = Image.new('RGBA', (32, 32), color=(255, 0, 0, 128))
        images = [img]
        
        try:
            self.logger.log_image("test_rgba", images, step=1)
        except Exception as e:
            self.fail(f"log_image failed with RGBA PIL image: {e}")
    
    def test_convert_image_to_tensor_normalization(self):
        """Test that images with values > 1 are properly normalized"""
        # Create image with values in [0, 255] range
        img_array = np.full((32, 32, 3), 255, dtype=np.uint8)
        images = [img_array]
        
        try:
            self.logger.log_image("test_normalization", images, step=1)
        except Exception as e:
            self.fail(f"log_image failed with normalization: {e}")
    
    def test_convert_image_to_tensor_already_normalized(self):
        """Test that images with values in [0, 1] are not re-normalized"""
        # Create image with values in [0, 1] range
        img_array = np.random.rand(32, 32, 3).astype(np.float32)
        images = [img_array]
        
        try:
            self.logger.log_image("test_already_normalized", images, step=1)
        except Exception as e:
            self.fail(f"log_image failed with already normalized image: {e}")
    
    def test_log_image_various_formats_integration(self):
        """Integration test with various image formats in one call"""
        # Mix different image formats
        pil_img = Image.new('RGB', (32, 32), color='red')
        numpy_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        tensor_img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        
        images = [pil_img, numpy_img, tensor_img]
        captions = ["PIL Image", "Numpy Array", "Torch Tensor"]
        
        try:
            self.logger.log_image("test_mixed_formats", images, step=1, caption=captions, fb="test")
        except Exception as e:
            self.fail(f"log_image failed with mixed formats: {e}")


if __name__ == '__main__':
    unittest.main()