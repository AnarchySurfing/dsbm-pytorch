#!/usr/bin/env python3
"""
Unit tests for TensorBoardLogger class
"""

import os
import tempfile
import shutil
import sys
from unittest.mock import patch, MagicMock

# Mock the dependencies that might not be available
class MockTensorBoardLogger:
    def __init__(self, **kwargs):
        self.log_dir = kwargs.get('save_dir', './logs')
        self.name = kwargs.get('name', 'test')
        self.version = kwargs.get('version', None)
        
    def log_metrics(self, metrics, step=None):
        pass
        
    def log_hyperparams(self, params):
        pass

# Try to import the real TensorBoardLogger, fall back to mock if dependencies missing
try:
    from bridge.runners.logger import TensorBoardLogger
except ImportError as e:
    print(f"Warning: Could not import TensorBoardLogger due to missing dependencies: {e}")
    print("Using mock for basic testing...")
    TensorBoardLogger = MockTensorBoardLogger


class TestTensorBoardLogger:
    """Test suite for TensorBoardLogger functionality"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = TensorBoardLogger(
            save_dir=self.temp_dir,
            name="test_experiment",
            version="test_version"
        )
    
    def teardown_method(self):
        """Clean up after each test method"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_logger_initialization(self):
        """Test that TensorBoardLogger initializes correctly"""
        assert self.logger is not None
        assert hasattr(self.logger, 'log_metrics')
        assert hasattr(self.logger, 'log_hyperparams')
        assert self.logger.LOGGER_JOIN_CHAR == '/'
    
    def test_log_metrics_basic(self):
        """Test basic metrics logging without fb parameter"""
        metrics = {
            'loss': 0.5,
            'accuracy': 0.95,
            'learning_rate': 0.001
        }
        
        # Should not raise any exceptions
        self.logger.log_metrics(metrics, step=1)
        
        # Verify log directory was created
        assert os.path.exists(self.logger.log_dir)
    
    def test_log_metrics_with_fb_parameter(self):
        """Test metrics logging with fb (forward/backward) prefixing"""
        metrics = {
            'loss': 0.5,
            'accuracy': 0.95,
            'fb': 'forward'  # This should be removed from metrics
        }
        
        # Test with fb in metrics dict
        self.logger.log_metrics(metrics, step=1, fb=None)
        
        # Test with fb as parameter
        metrics_no_fb = {'loss': 0.3, 'accuracy': 0.97}
        self.logger.log_metrics(metrics_no_fb, step=2, fb='backward')
        
        # Should not raise exceptions
        assert os.path.exists(self.logger.log_dir)
    
    def test_log_metrics_fb_prefixing(self):
        """Test that fb parameter correctly prefixes metric names"""
        # Mock the parent log_metrics to capture what gets passed
        with patch.object(self.logger.__class__.__bases__[0], 'log_metrics') as mock_log:
            metrics = {'loss': 0.5, 'accuracy': 0.95}
            self.logger.log_metrics(metrics, step=1, fb='forward')
            
            # Check that the parent log_metrics was called with prefixed metrics
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            logged_metrics = args[0] if args else kwargs['metrics']
            
            assert 'forward/loss' in logged_metrics
            assert 'forward/accuracy' in logged_metrics
            assert logged_metrics['forward/loss'] == 0.5
            assert logged_metrics['forward/accuracy'] == 0.95
    
    def test_log_hyperparams_basic(self):
        """Test basic hyperparameter logging"""
        params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'model_name': 'test_model'
        }
        
        # Should not raise any exceptions
        self.logger.log_hyperparams(params)
        assert os.path.exists(self.logger.log_dir)
    
    def test_log_hyperparams_nested_dict(self):
        """Test hyperparameter logging with nested dictionaries"""
        params = {
            'model': {
                'num_layers': 3,
                'hidden_size': 128,
                'activation': 'relu'
            },
            'optimizer': {
                'type': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0001
            },
            'training': {
                'batch_size': 32,
                'epochs': 100
            }
        }
        
        # Should not raise any exceptions
        self.logger.log_hyperparams(params)
        assert os.path.exists(self.logger.log_dir)
    
    def test_log_hyperparams_with_non_serializable_values(self):
        """Test hyperparameter logging handles non-serializable values gracefully"""
        params = {
            'learning_rate': 0.001,
            'model_object': torch.nn.Linear(10, 1),  # Non-serializable
            'none_value': None,
            'list_value': [1, 2, 3],
            'tuple_value': (4, 5, 6),
            'function': lambda x: x  # Non-serializable
        }
        
        # Should not raise any exceptions
        self.logger.log_hyperparams(params)
        assert os.path.exists(self.logger.log_dir)
    
    def test_flatten_dict(self):
        """Test the _flatten_dict helper method"""
        nested_dict = {
            'a': 1,
            'b': {
                'c': 2,
                'd': {
                    'e': 3
                }
            }
        }
        
        flattened = self.logger._flatten_dict(nested_dict)
        
        expected = {
            'a': 1,
            'b/c': 2,
            'b/d/e': 3
        }
        
        assert flattened == expected
    
    def test_flatten_dict_custom_separator(self):
        """Test _flatten_dict with custom separator"""
        nested_dict = {
            'a': 1,
            'b': {
                'c': 2
            }
        }
        
        flattened = self.logger._flatten_dict(nested_dict, sep='.')
        
        expected = {
            'a': 1,
            'b.c': 2
        }
        
        assert flattened == expected


def run_tests():
    """Run all tests"""
    import sys
    
    # Create test instance
    test_instance = TestTensorBoardLogger()
    
    # List of test methods
    test_methods = [
        'test_logger_initialization',
        'test_log_metrics_basic',
        'test_log_metrics_with_fb_parameter',
        'test_log_metrics_fb_prefixing',
        'test_log_hyperparams_basic',
        'test_log_hyperparams_nested_dict',
        'test_log_hyperparams_with_non_serializable_values',
        'test_flatten_dict',
        'test_flatten_dict_custom_separator'
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            print(f"Running {method_name}...")
            test_instance.setup_method()
            method = getattr(test_instance, method_name)
            method()
            test_instance.teardown_method()
            print(f"✓ {method_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name} FAILED: {e}")
            failed += 1
            test_instance.teardown_method()
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    run_tests()