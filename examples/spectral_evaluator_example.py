"""
Example usage of the SpectralEvaluator for visible-to-infrared image translation evaluation.

This script demonstrates how to use the SpectralEvaluator to assess the quality of
visible-to-infrared image translations using comprehensive metrics.
"""

import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from bridge.data.spectral_evaluator import SpectralEvaluator, create_spectral_evaluator


def create_synthetic_data(batch_size=8, image_size=128):
    """
    Create synthetic visible and infrared image pairs for demonstration.
    
    Args:
        batch_size: Number of image pairs to generate
        image_size: Size of the square images
        
    Returns:
        Tuple of (visible_images, target_infrared, generated_infrared)
    """
    print(f"Creating synthetic data: {batch_size} pairs of {image_size}x{image_size} images")
    
    # Create visible images with some structure (checkerboard pattern + noise)
    visible = torch.zeros(batch_size, 3, image_size, image_size)
    for i in range(batch_size):
        # Create checkerboard pattern
        checker = torch.zeros(image_size, image_size)
        checker[::16, ::16] = 1
        checker[8::16, 8::16] = 1
        
        # Add some random structure
        structure = torch.randn(image_size, image_size) * 0.3
        
        # Combine and add to all channels
        pattern = checker + structure
        visible[i] = pattern.unsqueeze(0).repeat(3, 1, 1)
    
    # Normalize to [-1, 1] range
    visible = (visible - visible.mean()) / visible.std() * 0.5
    visible = torch.clamp(visible, -1, 1)
    
    # Create target infrared images (simulate thermal characteristics)
    target_infrared = torch.zeros_like(visible)
    for i in range(batch_size):
        # Simulate thermal response (edges are cooler, centers warmer)
        thermal = torch.zeros(image_size, image_size)
        center_x, center_y = image_size // 2, image_size // 2
        
        for x in range(image_size):
            for y in range(image_size):
                dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                thermal[x, y] = 1.0 - (dist / (image_size * 0.7))
        
        # Add some noise and structure from visible
        thermal += 0.3 * torch.mean(visible[i], dim=0) + 0.1 * torch.randn(image_size, image_size)
        target_infrared[i] = thermal.unsqueeze(0).repeat(3, 1, 1)
    
    # Normalize target infrared
    target_infrared = (target_infrared - target_infrared.mean()) / target_infrared.std() * 0.5
    target_infrared = torch.clamp(target_infrared, -1, 1)
    
    # Create generated infrared (simulate model output with some error)
    generated_infrared = target_infrared + 0.1 * torch.randn_like(target_infrared)
    generated_infrared = torch.clamp(generated_infrared, -1, 1)
    
    return visible, target_infrared, generated_infrared


def demonstrate_basic_evaluation():
    """Demonstrate basic evaluation functionality."""
    print("\n" + "="*60)
    print("BASIC EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Create evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    evaluator = SpectralEvaluator(device=device)
    
    # Create synthetic data
    visible, target_ir, generated_ir = create_synthetic_data(batch_size=4, image_size=64)
    visible = visible.to(device)
    target_ir = target_ir.to(device)
    generated_ir = generated_ir.to(device)
    
    # Perform evaluation
    print("\nPerforming comprehensive evaluation...")
    results_dir = "spectral_evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results = evaluator.evaluate_batch(
        visible, target_ir, generated_ir,
        save_dir=results_dir,
        create_visualizations=True
    )
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"  FID Score: {results.fid_score:.4f}")
    print(f"  LPIPS Score: {results.lpips_score:.4f}")
    print(f"  PSNR Score: {results.psnr_score:.2f} dB")
    print(f"  SSIM Score: {results.ssim_score:.4f}")
    print(f"  Spectral Consistency: {results.spectral_consistency:.4f}")
    print(f"  Cross-Spectral Correlation: {results.cross_spectral_correlation:.4f}")
    
    print(f"\nVisualization files saved:")
    for plot_path in results.comparison_plots:
        print(f"  - {plot_path}")
    
    return results


def demonstrate_batch_processing():
    """Demonstrate processing multiple batches and tracking metrics over time."""
    print("\n" + "="*60)
    print("BATCH PROCESSING DEMONSTRATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = SpectralEvaluator(device=device)
    
    # Simulate multiple evaluation batches (e.g., during training)
    num_batches = 5
    metrics_history = {
        'fid': [],
        'lpips': [],
        'psnr': [],
        'ssim': [],
        'spectral_consistency': [],
        'cross_spectral_correlation': []
    }
    
    print(f"Processing {num_batches} batches...")
    
    for batch_idx in range(num_batches):
        print(f"\nBatch {batch_idx + 1}/{num_batches}")
        
        # Create data with gradually improving quality
        visible, target_ir, generated_ir = create_synthetic_data(batch_size=4, image_size=64)
        
        # Simulate improving model (reduce noise over time)
        noise_level = 0.2 * (1.0 - batch_idx / num_batches)
        generated_ir = target_ir + noise_level * torch.randn_like(target_ir)
        generated_ir = torch.clamp(generated_ir, -1, 1)
        
        visible = visible.to(device)
        target_ir = target_ir.to(device)
        generated_ir = generated_ir.to(device)
        
        # Evaluate batch
        results = evaluator.evaluate_batch(
            visible, target_ir, generated_ir,
            create_visualizations=False
        )
        
        # Store metrics
        metrics_history['fid'].append(results.fid_score)
        metrics_history['lpips'].append(results.lpips_score)
        metrics_history['psnr'].append(results.psnr_score)
        metrics_history['ssim'].append(results.ssim_score)
        metrics_history['spectral_consistency'].append(results.spectral_consistency)
        metrics_history['cross_spectral_correlation'].append(results.cross_spectral_correlation)
        
        print(f"  PSNR: {results.psnr_score:.2f} dB, SSIM: {results.ssim_score:.4f}")
    
    # Generate metrics visualization
    results_dir = "spectral_evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_plot = evaluator.generate_metrics_visualization(
        metrics_history, results_dir, "training_metrics_history.png"
    )
    
    print(f"\nMetrics history visualization saved: {metrics_plot}")
    
    # Print final metrics
    print(f"\nFinal Metrics (Batch {num_batches}):")
    print(f"  FID: {metrics_history['fid'][-1]:.4f}")
    print(f"  LPIPS: {metrics_history['lpips'][-1]:.4f}")
    print(f"  PSNR: {metrics_history['psnr'][-1]:.2f} dB")
    print(f"  SSIM: {metrics_history['ssim'][-1]:.4f}")
    print(f"  Spectral Consistency: {metrics_history['spectral_consistency'][-1]:.4f}")
    print(f"  Cross-Spectral Correlation: {metrics_history['cross_spectral_correlation'][-1]:.4f}")
    
    return metrics_history


def demonstrate_factory_function():
    """Demonstrate using the factory function for different configurations."""
    print("\n" + "="*60)
    print("FACTORY FUNCTION DEMONSTRATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create evaluator using factory function
    print("Creating evaluator with factory function...")
    evaluator = create_spectral_evaluator(
        device=device,
        dataset_name="demo_dataset",
        data_dir="./data"  # This path doesn't exist, but shows the pattern
    )
    
    print(f"Evaluator created successfully on device: {evaluator.device}")
    
    # Quick evaluation
    visible, target_ir, generated_ir = create_synthetic_data(batch_size=2, image_size=32)
    visible = visible.to(device)
    target_ir = target_ir.to(device)
    generated_ir = generated_ir.to(device)
    
    results = evaluator.evaluate_batch(
        visible, target_ir, generated_ir,
        create_visualizations=False
    )
    
    print(f"Quick evaluation - PSNR: {results.psnr_score:.2f} dB")


def demonstrate_individual_metrics():
    """Demonstrate using individual metric components."""
    print("\n" + "="*60)
    print("INDIVIDUAL METRICS DEMONSTRATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = SpectralEvaluator(device=device)
    
    # Create test data
    visible, target_ir, generated_ir = create_synthetic_data(batch_size=2, image_size=64)
    visible = visible.to(device)
    target_ir = target_ir.to(device)
    generated_ir = generated_ir.to(device)
    
    print("Testing individual metric components...")
    
    # Test spectral consistency metrics
    gradient_consistency = evaluator.spectral_consistency.compute_gradient_consistency(
        visible, generated_ir
    )
    structural_consistency = evaluator.spectral_consistency.compute_structural_consistency(
        visible, generated_ir
    )
    
    print(f"Gradient Consistency: {gradient_consistency:.4f}")
    print(f"Structural Consistency: {structural_consistency:.4f}")
    
    # Test cross-spectral correlation metrics
    freq_correlation = evaluator.cross_spectral_correlation.compute_frequency_domain_correlation(
        visible, generated_ir
    )
    mutual_info = evaluator.cross_spectral_correlation.compute_mutual_information(
        visible, generated_ir
    )
    
    print(f"Frequency Domain Correlation: {freq_correlation:.4f}")
    print(f"Mutual Information: {mutual_info:.4f}")
    
    # Test standard metrics update
    evaluator.update_metrics(generated_ir, target_ir, visible)
    
    print(f"PSNR: {evaluator.psnr.compute():.2f} dB")
    print(f"SSIM: {evaluator.ssim.compute():.4f}")


def main():
    """Run all demonstrations."""
    print("SpectralEvaluator Demonstration")
    print("This script shows how to use the SpectralEvaluator for visible-to-infrared translation evaluation.")
    
    try:
        # Run demonstrations
        demonstrate_basic_evaluation()
        demonstrate_batch_processing()
        demonstrate_factory_function()
        demonstrate_individual_metrics()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nCheck the 'spectral_evaluation_results' directory for generated visualizations.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()