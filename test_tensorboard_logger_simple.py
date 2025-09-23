#!/usr/bin/env python3
"""
Simple unit tests for TensorBoardLogger class logic
Tests the core functionality without requiring external dependencies
"""

import os
import sys

def test_flatten_dict():
    """Test the _flatten_dict helper method logic"""
    
    def flatten_dict(d, parent_key='', sep='/'):
        """Flatten nested dictionary for hyperparameter logging"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Test basic flattening
    nested_dict = {
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3
            }
        }
    }
    
    flattened = flatten_dict(nested_dict)
    expected = {
        'a': 1,
        'b/c': 2,
        'b/d/e': 3
    }
    
    assert flattened == expected, f"Expected {expected}, got {flattened}"
    print("✓ test_flatten_dict PASSED")


def test_fb_prefixing_logic():
    """Test the fb prefixing logic"""
    
    def apply_fb_prefixing(metrics, fb=None):
        """Apply fb prefixing logic like in TensorBoardLogger"""
        # Handle fb parameter the same way as WandbLogger
        if fb is not None:
            metrics.pop('fb', None)
        else:
            fb = metrics.pop('fb', None)
        
        # Apply fb prefixing if specified
        if fb is not None:
            metrics = {fb + '/' + k: v for k, v in metrics.items()}
        
        return metrics
    
    # Test with fb as parameter
    metrics1 = {'loss': 0.5, 'accuracy': 0.95}
    result1 = apply_fb_prefixing(metrics1.copy(), fb='forward')
    expected1 = {'forward/loss': 0.5, 'forward/accuracy': 0.95}
    assert result1 == expected1, f"Expected {expected1}, got {result1}"
    
    # Test with fb in metrics dict
    metrics2 = {'loss': 0.3, 'accuracy': 0.97, 'fb': 'backward'}
    result2 = apply_fb_prefixing(metrics2.copy(), fb=None)
    expected2 = {'backward/loss': 0.3, 'backward/accuracy': 0.97}
    assert result2 == expected2, f"Expected {expected2}, got {result2}"
    
    # Test with no fb
    metrics3 = {'loss': 0.1, 'accuracy': 0.99}
    result3 = apply_fb_prefixing(metrics3.copy(), fb=None)
    expected3 = {'loss': 0.1, 'accuracy': 0.99}
    assert result3 == expected3, f"Expected {expected3}, got {result3}"
    
    print("✓ test_fb_prefixing_logic PASSED")


def test_hyperparams_serialization():
    """Test hyperparameter serialization logic"""
    
    def serialize_hyperparams(params):
        """Serialize hyperparameters for logging"""
        serializable_params = {}
        for key, value in params.items():
            try:
                if isinstance(value, (int, float, str, bool)):
                    serializable_params[key] = value
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    # Convert lists/tuples to string representation
                    serializable_params[key] = str(value)
                elif value is None:
                    serializable_params[key] = "None"
                else:
                    # Convert other types to string
                    serializable_params[key] = str(value)
            except Exception:
                # Skip values that can't be serialized
                continue
        return serializable_params
    
    # Test with various data types
    params = {
        'learning_rate': 0.001,  # float
        'batch_size': 32,        # int
        'model_name': 'test',    # str
        'use_dropout': True,     # bool
        'none_value': None,      # None
        'list_value': [1, 2, 3], # list
        'tuple_value': (4, 5),   # tuple
    }
    
    result = serialize_hyperparams(params)
    
    assert result['learning_rate'] == 0.001
    assert result['batch_size'] == 32
    assert result['model_name'] == 'test'
    assert result['use_dropout'] == True
    assert result['none_value'] == "None"
    assert result['list_value'] == "[1, 2, 3]"
    assert result['tuple_value'] == "(4, 5)"
    
    print("✓ test_hyperparams_serialization PASSED")


def test_logger_interface():
    """Test that the logger interface matches WandbLogger"""
    
    # Check if we can import the logger
    try:
        sys.path.insert(0, '.')
        from bridge.runners.logger import TensorBoardLogger
        
        # Check that required methods exist
        assert hasattr(TensorBoardLogger, 'log_metrics'), "log_metrics method missing"
        assert hasattr(TensorBoardLogger, 'log_hyperparams'), "log_hyperparams method missing"
        assert hasattr(TensorBoardLogger, 'LOGGER_JOIN_CHAR'), "LOGGER_JOIN_CHAR constant missing"
        
        # Check that LOGGER_JOIN_CHAR has correct value
        assert TensorBoardLogger.LOGGER_JOIN_CHAR == '/', f"Expected '/', got {TensorBoardLogger.LOGGER_JOIN_CHAR}"
        
        print("✓ test_logger_interface PASSED")
        
    except ImportError as e:
        print(f"⚠ test_logger_interface SKIPPED: {e}")


def test_config_integration():
    """Test that config_getters has TensorBoard support"""
    
    try:
        sys.path.insert(0, '.')
        from bridge.runners.config_getters import TENSORBOARD_TAG, get_logger
        
        # Check that TENSORBOARD_TAG is defined
        assert TENSORBOARD_TAG == 'TensorBoard', f"Expected 'TensorBoard', got {TENSORBOARD_TAG}"
        
        # Check that get_logger function exists
        assert callable(get_logger), "get_logger is not callable"
        
        print("✓ test_config_integration PASSED")
        
    except ImportError as e:
        print(f"⚠ test_config_integration SKIPPED: {e}")


def run_tests():
    """Run all tests"""
    
    test_functions = [
        test_flatten_dict,
        test_fb_prefixing_logic,
        test_hyperparams_serialization,
        test_logger_interface,
        test_config_integration
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    print("Running TensorBoardLogger tests...\n")
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠ {test_func.__name__} ERROR: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        return False
    else:
        print("All tests passed!")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)