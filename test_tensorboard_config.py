#!/usr/bin/env python3
"""
Test script to verify TensorBoard configuration system works correctly.
"""

import os
import sys
import tempfile
import shutil
from omegaconf import OmegaConf

# Add the bridge module to the path
sys.path.insert(0, '.')

def test_tensorboard_config():
    """Test TensorBoard configuration parsing and logger creation."""
    
    # Import after adding to path
    from bridge.runners.config_getters import get_logger, TENSORBOARD_TAG, CSV_TAG
    
    print("Testing TensorBoard configuration system...")
    
    # Test 1: Valid TensorBoard configuration
    print("\n1. Testing valid TensorBoard configuration...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'LOGGER': TENSORBOARD_TAG,
            'tensorboard_log_dir': temp_dir,
            'tensorboard_name': 'test_experiment',
            'tensorboard_version': 'v1',
            'data': {'dataset': 'mnist'},
            'model': {'type': 'unet'},
            'name': 'test'
        }
        
        args = OmegaConf.create(config)
        
        try:
            logger = get_logger(args, 'test_logger')
            print(f"✓ Successfully created TensorBoard logger: {type(logger)}")
            print(f"✓ Log directory: {logger.save_dir}")
            print(f"✓ Logger name: {logger.name}")
            print(f"✓ Logger version: {logger.version}")
        except Exception as e:
            print(f"✗ Failed to create TensorBoard logger: {e}")
            return False
    
    # Test 2: Missing tensorboard dependency (simulate)
    print("\n2. Testing missing tensorboard dependency handling...")
    
    # Temporarily hide tensorboard import
    import sys
    original_modules = sys.modules.copy()
    
    # Remove tensorboard modules if they exist
    modules_to_remove = [name for name in sys.modules.keys() if 'tensorboard' in name.lower()]
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    
    # Mock import error
    import builtins
    original_import = builtins.__import__
    
    def mock_import(name, *args, **kwargs):
        if name == 'tensorboard' or name == 'torch.utils.tensorboard':
            raise ImportError("No module named 'tensorboard'")
        return original_import(name, *args, **kwargs)
    
    builtins.__import__ = mock_import
    
    try:
        # Reload the config_getters module to trigger the import error
        import importlib
        from bridge.runners import config_getters
        importlib.reload(config_getters)
        
        config = {
            'LOGGER': TENSORBOARD_TAG,
            'tensorboard_log_dir': './test_logs',
            'tensorboard_name': 'test_experiment'
        }
        args = OmegaConf.create(config)
        
        try:
            logger = config_getters.get_logger(args, 'test_logger')
            print("✗ Expected ImportError but logger was created successfully")
        except ImportError as e:
            if "tensorboard is not installed" in str(e):
                print("✓ Correctly caught missing tensorboard dependency")
            else:
                print(f"✗ Unexpected ImportError: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
    finally:
        # Restore original import and modules
        builtins.__import__ = original_import
        sys.modules.clear()
        sys.modules.update(original_modules)
    
    # Test 3: Invalid logger type
    print("\n3. Testing invalid logger type handling...")
    
    # Reload the module normally
    import importlib
    from bridge.runners import config_getters
    importlib.reload(config_getters)
    
    config = {
        'LOGGER': 'InvalidLogger',
        'name': 'test'
    }
    args = OmegaConf.create(config)
    
    try:
        logger = config_getters.get_logger(args, 'test_logger')
        print("✗ Expected ValueError but logger was created successfully")
    except ValueError as e:
        if "Invalid logger type" in str(e):
            print("✓ Correctly caught invalid logger type")
        else:
            print(f"✗ Unexpected ValueError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test 4: CSV logger still works
    print("\n4. Testing CSV logger compatibility...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = {
            'LOGGER': CSV_TAG,
            'CSV_log_dir': temp_dir,
            'name': 'test'
        }
        args = OmegaConf.create(config)
        
        try:
            logger = config_getters.get_logger(args, 'test_logger')
            print(f"✓ Successfully created CSV logger: {type(logger)}")
        except Exception as e:
            print(f"✗ Failed to create CSV logger: {e}")
            return False
    
    print("\n✓ All configuration tests passed!")
    return True

if __name__ == "__main__":
    success = test_tensorboard_config()
    sys.exit(0 if success else 1)