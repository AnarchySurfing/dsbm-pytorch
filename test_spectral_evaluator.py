"""
Unit tests for the SpectralEvaluator module.

Tests all components of the spectral evaluation system including:
- Standard image quality metrics (FID, LPIPS, PSNR, SSIM)
- Spectral consistency metrics
- Cross-spectral correlation metrics
- Visualization functions
"""

import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import the spectral evaluator components
from bridge.data.spectral_evaluator import (
    SpectralEvaluator, 
    SpectralConsistencyMetric, 
    CrossSpectralCorrelationMetric,
    SpectralEvaluationResults,
    create_spectral_evaluator
)


class TestSpectralConsistencyMetric:
    """Test the SpectralConsistencyMetric class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'  # Use CPU for testing
        self.metric = SpectralConsistencyMetric(device=self.device)
        
        # Create test images
        self.batch_size = 2
        self.channels = 3
        self.height = 64
        self.width = 64
        
        # Create synthetic visible and infrared images
        self.visible = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.infrared = torch.randn(self.batch_size, self.channels, self.height, self.width)
        
        # Create similar infrared (should have high consistency)
        self.similar_infrared = self.visible + 0.1 * torch.randn_like(self.visible)
    
    def test_gradient_consistency_computation(self):
        """Test gradient consistency computation."""
        # Test with similar images (should have high consistency)
        consistency_high = self.metric.compute_gradient_consistency(self.visible, self.similar_infrared)
        
        # Test with random images (should have lower consistency)
        consistency_low = self.metric.compute_gradient_consistency(self.visible, self.infrared)
        
        # Assertions
        assert isinstance(consistency_high, float)
        assert isinstance(consistency_low, float)
        assert -1 <= consistency_high <= 1
        assert -1 <= consistency_low <= 1
        assert consistency_high > consistency_low  # Similar images should have higher consistency
    
    def test_structural_consistency_computation(self):
        """Test structural consistency computation."""
        # Test with similar images
        consistency_high = self.metric.compute_structural_consistency(self.visible, self.similar_infrared)
        
        # Test with random images
        consistency_low = self.metric.compute_structural_consistency(self.visible, self.infrared)
        
        # Assertions
        assert isinstance(consistency_high, float)
        assert isinstance(consistency_low, float)
        assert -1 <= consistency_high <= 1
        assert -1 <= consistency_low <= 1
    
    def test_edge_cases(self):
        """Test edge cases for spectral consistency."""
        # Test with identical images (should have perfect consistency)
        perfect_consistency = self.metric.compute_gradient_consistency(self.visible, self.visible)
        assert perfect_consistency > 0.9  # Should be very high
        
        # Test with single pixel images
        single_pixel = torch.ones(1, 3, 1, 1)
        consistency = self.metric.compute_gradient_consistency(single_pixel, single_pixel)
        assert isinstance(consistency, float)


class TestCrossSpectralCorrelationMetric:
    """Test the CrossSpectralCorrelationMetric class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.metric = CrossSpectralCorrelationMetric(device=self.device)
        
        # Create test images
        self.batch_size = 2
        self.channels = 3
        self.height = 32  # Smaller for FFT tests
        self.width = 32
        
        self.visible = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.infrared = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.similar_infrared = self.visible + 0.1 * torch.randn_like(self.visible)
    
    def test_frequency_domain_correlation(self):
        """Test frequency domain correlation computation."""
        # Test with similar images
        correlation_high = self.metric.compute_frequency_domain_correlation(self.visible, self.similar_infrared)
        
        # Test with random images
        correlation_low = self.metric.compute_frequency_domain_correlation(self.visible, self.infrared)
        
        # Assertions
        assert isinstance(correlation_high, float)
        assert isinstance(correlation_low, float)
        assert -1 <= correlation_high <= 1
        assert -1 <= correlation_low <= 1
        assert correlation_high > correlation_low
    
    def test_mutual_information(self):
        """Test mutual information computation."""
        # Test with similar images
        mi_high = self.metric.compute_mutual_information(self.visible, self.similar_infrared)
        
        # Test with random images
        mi_low = self.metric.compute_mutual_information(self.visible, self.infrared)
        
        # Assertions
        assert isinstance(mi_high, float)
        assert isinstance(mi_low, float)
        assert mi_high >= 0  # MI is non-negative
        assert mi_low >= 0
        assert mi_high > mi_low  # Similar images should have higher MI
    
    def test_mutual_information_bins(self):
        """Test mutual information with different bin sizes."""
        mi_32 = self.metric.compute_mutual_information(self.visible, self.similar_infrared, bins=32)
        mi_64 = self.metric.compute_mutual_information(self.visible, self.similar_infrared, bins=64)
        
        assert isinstance(mi_32, float)
        assert isinstance(mi_64, float)
        assert mi_32 >= 0
        assert mi_64 >= 0


class TestSpectralEvaluator:
    """Test the main SpectralEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.evaluator = SpectralEvaluator(device=self.device)
        
        # Create test data
        self.batch_size = 4
        self.channels = 3
        self.height = 64
        self.width = 64
        
        # Create test images in [-1, 1] range (as expected by the model)
        self.visible = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.target_ir = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.generated_ir = self.target_ir + 0.1 * torch.randn_like(self.target_ir)  # Similar to target
        
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test SpectralEvaluator initialization."""
        assert self.evaluator.device == self.device
        assert hasattr(self.evaluator, 'psnr')
        assert hasattr(self.evaluator, 'ssim')
        assert hasattr(self.evaluator, 'fid')
        assert hasattr(self.evaluator, 'lpips')
        assert hasattr(self.evaluator, 'spectral_consistency')
        assert hasattr(self.evaluator, 'cross_spectral_correlation')
    
    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        # Update metrics first
        self.evaluator.update_metrics(self.generated_ir, self.target_ir, self.visible)
        
        # Reset metrics
        self.evaluator.reset_metrics()
        
        # This should not raise an error
        assert True
    
    def test_update_metrics(self):
        """Test metrics update functionality."""
        # This should not raise an error
        self.evaluator.update_metrics(self.generated_ir, self.target_ir, self.visible)
        assert True
    
    def test_compute_cross_spectral_metrics(self):
        """Test comprehensive metrics computation."""
        # Update metrics first
        self.evaluator.update_metrics(self.generated_ir, self.target_ir, self.visible)
        
        # Compute metrics
        metrics = self.evaluator.compute_cross_spectral_metrics(
            self.generated_ir, self.target_ir, self.visible
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['fid', 'lpips', 'psnr', 'ssim', 'spectral_consistency', 'cross_spectral_correlation']
        for metric_name in expected_metrics:
            assert metric_name in metrics
            assert isinstance(metrics[metric_name], float)
        
        # Check reasonable ranges
        assert metrics['psnr'] > 0  # PSNR should be positive
        assert 0 <= metrics['ssim'] <= 1  # SSIM should be in [0, 1]
        assert metrics['fid'] >= 0  # FID should be non-negative
        assert metrics['lpips'] >= 0  # LPIPS should be non-negative
    
    def test_generate_comparison_plots(self):
        """Test comparison plot generation."""
        plots = self.evaluator.generate_comparison_plots(
            self.visible, self.target_ir, self.generated_ir, 
            self.temp_dir, num_samples=2
        )
        
        # Check that plots were created
        assert len(plots) == 2  # Grid plot and difference plot
        for plot_path in plots:
            assert os.path.exists(plot_path)
            assert plot_path.endswith('.png')
    
    def test_generate_metrics_visualization(self):
        """Test metrics history visualization."""
        # Create fake metrics history
        metrics_history = {
            'fid': [10.0, 8.0, 6.0, 5.0],
            'lpips': [0.5, 0.4, 0.3, 0.25],
            'psnr': [20.0, 22.0, 24.0, 25.0],
            'ssim': [0.6, 0.7, 0.8, 0.85],
            'spectral_consistency': [0.5, 0.6, 0.7, 0.75],
            'cross_spectral_correlation': [0.4, 0.5, 0.6, 0.65]
        }
        
        plot_path = self.evaluator.generate_metrics_visualization(
            metrics_history, self.temp_dir
        )
        
        assert os.path.exists(plot_path)
        assert plot_path.endswith('.png')
    
    def test_evaluate_batch(self):
        """Test comprehensive batch evaluation."""
        results = self.evaluator.evaluate_batch(
            self.visible, self.target_ir, self.generated_ir,
            save_dir=self.temp_dir, create_visualizations=True
        )
        
        # Check results structure
        assert isinstance(results, SpectralEvaluationResults)
        assert isinstance(results.fid_score, float)
        assert isinstance(results.lpips_score, float)
        assert isinstance(results.psnr_score, float)
        assert isinstance(results.ssim_score, float)
        assert isinstance(results.spectral_consistency, float)
        assert isinstance(results.cross_spectral_correlation, float)
        assert len(results.visual_samples) == 3  # visible, target, generated
        assert len(results.comparison_plots) == 2  # grid and difference plots
        
        # Check that visualization files exist
        for plot_path in results.comparison_plots:
            assert os.path.exists(plot_path)
    
    def test_evaluate_batch_no_visualizations(self):
        """Test batch evaluation without visualizations."""
        results = self.evaluator.evaluate_batch(
            self.visible, self.target_ir, self.generated_ir,
            create_visualizations=False
        )
        
        assert isinstance(results, SpectralEvaluationResults)
        assert len(results.visual_samples) == 0
        assert len(results.comparison_plots) == 0


class TestFactoryFunction:
    """Test the factory function for creating SpectralEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_spectral_evaluator_basic(self):
        """Test basic evaluator creation."""
        evaluator = create_spectral_evaluator(device='cpu')
        assert isinstance(evaluator, SpectralEvaluator)
        assert evaluator.device == 'cpu'
    
    def test_create_spectral_evaluator_with_fid_stats(self):
        """Test evaluator creation with FID statistics."""
        # Create fake FID stats file
        fid_stats = {
            'real_features_sum': torch.zeros(2048),
            'real_features_cov_sum': torch.zeros(2048, 2048),
            'real_features_num_samples': torch.tensor(1000)
        }
        
        fid_stats_path = os.path.join(self.temp_dir, 'test_dataset', 'fid_stats.pt')
        os.makedirs(os.path.dirname(fid_stats_path), exist_ok=True)
        torch.save(fid_stats, fid_stats_path)
        
        evaluator = create_spectral_evaluator(
            device='cpu', 
            dataset_name='test_dataset', 
            data_dir=self.temp_dir
        )
        
        assert isinstance(evaluator, SpectralEvaluator)
        assert not evaluator.fid.reset_real_features  # Should be False when stats are loaded


def test_integration():
    """Integration test for the complete spectral evaluation pipeline."""
    device = 'cpu'
    evaluator = SpectralEvaluator(device=device)
    
    # Create test data
    batch_size = 2
    visible = torch.randn(batch_size, 3, 32, 32)
    target_ir = torch.randn(batch_size, 3, 32, 32)
    generated_ir = target_ir + 0.05 * torch.randn_like(target_ir)
    
    # Run complete evaluation
    with tempfile.TemporaryDirectory() as temp_dir:
        results = evaluator.evaluate_batch(
            visible, target_ir, generated_ir,
            save_dir=temp_dir, create_visualizations=True
        )
        
        # Verify all components work together
        assert isinstance(results, SpectralEvaluationResults)
        assert results.psnr_score > 0
        assert 0 <= results.ssim_score <= 1
        assert results.fid_score >= 0
        assert results.lpips_score >= 0
        assert len(results.comparison_plots) == 2
        
        # Verify files were created
        for plot_path in results.comparison_plots:
            assert os.path.exists(plot_path)


if __name__ == "__main__":
    # Run basic tests
    print("Running SpectralEvaluator tests...")
    
    # Test spectral consistency metric
    print("Testing SpectralConsistencyMetric...")
    test_consistency = TestSpectralConsistencyMetric()
    test_consistency.setup_method()
    test_consistency.test_gradient_consistency_computation()
    test_consistency.test_structural_consistency_computation()
    test_consistency.test_edge_cases()
    print("✓ SpectralConsistencyMetric tests passed")
    
    # Test cross-spectral correlation metric
    print("Testing CrossSpectralCorrelationMetric...")
    test_correlation = TestCrossSpectralCorrelationMetric()
    test_correlation.setup_method()
    test_correlation.test_frequency_domain_correlation()
    test_correlation.test_mutual_information()
    test_correlation.test_mutual_information_bins()
    print("✓ CrossSpectralCorrelationMetric tests passed")
    
    # Test main evaluator
    print("Testing SpectralEvaluator...")
    test_evaluator = TestSpectralEvaluator()
    test_evaluator.setup_method()
    test_evaluator.test_initialization()
    test_evaluator.test_reset_metrics()
    test_evaluator.test_update_metrics()
    test_evaluator.test_compute_cross_spectral_metrics()
    test_evaluator.test_generate_comparison_plots()
    test_evaluator.test_generate_metrics_visualization()
    test_evaluator.test_evaluate_batch()
    test_evaluator.test_evaluate_batch_no_visualizations()
    test_evaluator.teardown_method()
    print("✓ SpectralEvaluator tests passed")
    
    # Test factory function
    print("Testing factory function...")
    test_factory = TestFactoryFunction()
    test_factory.setup_method()
    test_factory.test_create_spectral_evaluator_basic()
    test_factory.test_create_spectral_evaluator_with_fid_stats()
    test_factory.teardown_method()
    print("✓ Factory function tests passed")
    
    # Run integration test
    print("Running integration test...")
    test_integration()
    print("✓ Integration test passed")
    
    print("\nAll tests passed! ✓")