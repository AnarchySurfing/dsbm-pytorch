"""
Spectral Evaluation Module for Visible-to-Infrared Image Translation

This module provides comprehensive evaluation capabilities for cross-spectral image translation,
including standard image quality metrics (FID, LPIPS, PSNR, SSIM) and spectral consistency metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import seaborn as sns

from .metrics import PSNR, SSIM, FID, LPIPS
from .utils import normalize_tensor, to_uint8_tensor, save_image


@dataclass
class SpectralEvaluationResults:
    """Container for spectral evaluation results"""
    fid_score: float
    lpips_score: float
    psnr_score: float
    ssim_score: float
    spectral_consistency: float
    cross_spectral_correlation: float
    visual_samples: List[torch.Tensor]
    comparison_plots: List[str]  # File paths to saved plots


class SpectralConsistencyMetric:
    """
    Computes spectral consistency between visible and infrared images.
    Measures how well the generated infrared preserves structural information from visible.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def compute_gradient_consistency(self, visible: torch.Tensor, infrared: torch.Tensor) -> float:
        """
        Compute gradient consistency between visible and infrared images.
        Measures how well edge information is preserved across spectral domains.
        """
        # Compute gradients using Sobel operators
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        
        # Convert to grayscale for gradient computation
        visible_gray = torch.mean(visible, dim=1, keepdim=True)
        infrared_gray = torch.mean(infrared, dim=1, keepdim=True)
        
        # Compute gradients
        vis_grad_x = F.conv2d(visible_gray, sobel_x, padding=1)
        vis_grad_y = F.conv2d(visible_gray, sobel_y, padding=1)
        ir_grad_x = F.conv2d(infrared_gray, sobel_x, padding=1)
        ir_grad_y = F.conv2d(infrared_gray, sobel_y, padding=1)
        
        # Compute gradient magnitudes
        vis_grad_mag = torch.sqrt(vis_grad_x**2 + vis_grad_y**2)
        ir_grad_mag = torch.sqrt(ir_grad_x**2 + ir_grad_y**2)
        
        # Normalize gradients
        vis_grad_norm = F.normalize(vis_grad_mag.flatten(1), dim=1)
        ir_grad_norm = F.normalize(ir_grad_mag.flatten(1), dim=1)
        
        # Compute cosine similarity
        consistency = F.cosine_similarity(vis_grad_norm, ir_grad_norm, dim=1)
        return consistency.mean().item()
    
    def compute_structural_consistency(self, visible: torch.Tensor, infrared: torch.Tensor) -> float:
        """
        Compute structural consistency using local binary patterns.
        Measures preservation of local texture patterns across spectral domains.
        """
        def local_binary_pattern(image: torch.Tensor) -> torch.Tensor:
            """Simplified LBP computation"""
            # Convert to grayscale
            gray = torch.mean(image, dim=1, keepdim=True)
            
            # Define 3x3 neighborhood offsets
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
            
            # Pad image
            padded = F.pad(gray, (1, 1, 1, 1), mode='reflect')
            
            lbp = torch.zeros_like(gray)
            for i, (dy, dx) in enumerate(offsets):
                neighbor = padded[:, :, 1+dy:padded.shape[2]-1+dy, 1+dx:padded.shape[3]-1+dx]
                lbp += (neighbor >= gray).float() * (2 ** i)
            
            return lbp
        
        vis_lbp = local_binary_pattern(visible)
        ir_lbp = local_binary_pattern(infrared)
        
        # Compute histogram correlation
        vis_hist = torch.histc(vis_lbp.flatten(), bins=256, min=0, max=255)
        ir_hist = torch.histc(ir_lbp.flatten(), bins=256, min=0, max=255)
        
        # Normalize histograms
        vis_hist = vis_hist / vis_hist.sum()
        ir_hist = ir_hist / ir_hist.sum()
        
        # Compute correlation coefficient
        correlation = F.cosine_similarity(vis_hist.unsqueeze(0), ir_hist.unsqueeze(0), dim=1)
        return correlation.item()


class CrossSpectralCorrelationMetric:
    """
    Computes cross-spectral correlation metrics between visible and infrared domains.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute_frequency_domain_correlation(self, visible: torch.Tensor, infrared: torch.Tensor) -> float:
        """
        Compute correlation in frequency domain using FFT.
        Measures how well frequency components are preserved across spectral domains.
        """
        # Convert to grayscale and compute FFT
        vis_gray = torch.mean(visible, dim=1)
        ir_gray = torch.mean(infrared, dim=1)
        
        # Compute 2D FFT
        vis_fft = torch.fft.fft2(vis_gray)
        ir_fft = torch.fft.fft2(ir_gray)
        
        # Compute magnitude spectra
        vis_mag = torch.abs(vis_fft)
        ir_mag = torch.abs(ir_fft)
        
        # Flatten and normalize
        vis_mag_flat = F.normalize(vis_mag.flatten(1), dim=1)
        ir_mag_flat = F.normalize(ir_mag.flatten(1), dim=1)
        
        # Compute correlation
        correlation = F.cosine_similarity(vis_mag_flat, ir_mag_flat, dim=1)
        return correlation.mean().item()
    
    def compute_mutual_information(self, visible: torch.Tensor, infrared: torch.Tensor, bins: int = 64) -> float:
        """
        Compute mutual information between visible and infrared images.
        Measures statistical dependence between spectral domains.
        """
        # Convert to grayscale and normalize to [0, 1]
        vis_gray = torch.mean(visible, dim=1).flatten()
        ir_gray = torch.mean(infrared, dim=1).flatten()
        
        vis_norm = (vis_gray - vis_gray.min()) / (vis_gray.max() - vis_gray.min() + 1e-8)
        ir_norm = (ir_gray - ir_gray.min()) / (ir_gray.max() - ir_gray.min() + 1e-8)
        
        # Discretize to bins
        vis_discrete = (vis_norm * (bins - 1)).long()
        ir_discrete = (ir_norm * (bins - 1)).long()
        
        # Compute joint histogram
        joint_hist = torch.zeros(bins, bins, device=self.device)
        for i in range(len(vis_discrete)):
            joint_hist[vis_discrete[i], ir_discrete[i]] += 1
        
        # Normalize to probabilities
        joint_prob = joint_hist / joint_hist.sum()
        
        # Compute marginal probabilities
        vis_prob = joint_prob.sum(dim=1)
        ir_prob = joint_prob.sum(dim=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * torch.log(joint_prob[i, j] / (vis_prob[i] * ir_prob[j] + 1e-8))
        
        return mi.item()


class SpectralEvaluator:
    """
    Comprehensive evaluation module for visible-to-infrared image translation.
    
    Provides standard image quality metrics (FID, LPIPS, PSNR, SSIM) adapted for
    cross-spectral evaluation, along with specialized spectral consistency metrics.
    """
    
    def __init__(self, device: str = 'cuda', fid_features_path: Optional[str] = None):
        """
        Initialize the SpectralEvaluator.
        
        Args:
            device: Device to run computations on ('cuda' or 'cpu')
            fid_features_path: Path to precomputed FID features for the dataset
        """
        self.device = device
        
        # Initialize standard metrics
        self.psnr = PSNR().to(device)
        self.ssim = SSIM().to(device)
        self.fid = FID().to(device)
        self.lpips = LPIPS().to(device)
        
        # Initialize spectral-specific metrics
        self.spectral_consistency = SpectralConsistencyMetric(device)
        self.cross_spectral_correlation = CrossSpectralCorrelationMetric(device)
        
        # Load precomputed FID features if available
        if fid_features_path and os.path.exists(fid_features_path):
            self._load_fid_features(fid_features_path)
    
    def _load_fid_features(self, features_path: str):
        """Load precomputed FID features for the dataset."""
        fid_stats = torch.load(features_path, map_location=self.device)
        self.fid.real_features_sum = fid_stats["real_features_sum"].to(self.device)
        self.fid.real_features_cov_sum = fid_stats["real_features_cov_sum"].to(self.device)
        self.fid.real_features_num_samples = fid_stats["real_features_num_samples"].to(self.device)
        self.fid.reset_real_features = False
    
    def reset_metrics(self):
        """Reset all metrics for a new evaluation."""
        self.psnr.reset()
        self.ssim.reset()
        self.fid.reset()
        self.lpips.reset()
    
    def update_metrics(self, generated_ir: torch.Tensor, target_ir: torch.Tensor, 
                      visible_input: torch.Tensor):
        """
        Update metrics with a batch of images.
        
        Args:
            generated_ir: Generated infrared images [B, C, H, W]
            target_ir: Target infrared images [B, C, H, W]
            visible_input: Input visible images [B, C, H, W]
        """
        # Convert to uint8 for standard metrics
        generated_uint8 = to_uint8_tensor(generated_ir)
        target_uint8 = to_uint8_tensor(target_ir)
        
        # Update standard metrics
        self.psnr.update(generated_uint8, target_uint8)
        self.ssim.update(generated_uint8, target_uint8)
        self.fid.update(generated_uint8, target_uint8)
        self.lpips.update(generated_uint8, target_uint8)
    
    def compute_cross_spectral_metrics(self, generated_ir: torch.Tensor, target_ir: torch.Tensor, 
                                     visible_input: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive cross-spectral metrics.
        
        Args:
            generated_ir: Generated infrared images [B, C, H, W]
            target_ir: Target infrared images [B, C, H, W]
            visible_input: Input visible images [B, C, H, W]
            
        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Standard image quality metrics
        metrics['fid'] = self.fid.compute().item()
        metrics['lpips'] = self.lpips.compute().item()
        metrics['psnr'] = self.psnr.compute().item()
        metrics['ssim'] = self.ssim.compute().item()
        
        # Spectral consistency metrics
        gradient_consistency = self.spectral_consistency.compute_gradient_consistency(
            visible_input, generated_ir
        )
        structural_consistency = self.spectral_consistency.compute_structural_consistency(
            visible_input, generated_ir
        )
        metrics['spectral_consistency'] = (gradient_consistency + structural_consistency) / 2
        
        # Cross-spectral correlation metrics
        freq_correlation = self.cross_spectral_correlation.compute_frequency_domain_correlation(
            visible_input, generated_ir
        )
        mutual_info = self.cross_spectral_correlation.compute_mutual_information(
            visible_input, generated_ir
        )
        metrics['cross_spectral_correlation'] = (freq_correlation + mutual_info) / 2
        
        return metrics
    
    def generate_comparison_plots(self, visible: torch.Tensor, target_ir: torch.Tensor, 
                                generated_ir: torch.Tensor, save_dir: str, 
                                filename_prefix: str = "spectral_comparison",
                                num_samples: int = 8) -> List[str]:
        """
        Generate side-by-side comparison plots of visible, target infrared, and generated infrared images.
        
        Args:
            visible: Input visible images [B, C, H, W]
            target_ir: Target infrared images [B, C, H, W]
            generated_ir: Generated infrared images [B, C, H, W]
            save_dir: Directory to save plots
            filename_prefix: Prefix for saved plot filenames
            num_samples: Number of samples to include in plots
            
        Returns:
            List of file paths to saved plots
        """
        os.makedirs(save_dir, exist_ok=True)
        saved_plots = []
        
        # Ensure we don't exceed available samples
        actual_samples = min(num_samples, visible.shape[0])
        
        # Normalize tensors for visualization
        visible_norm = normalize_tensor(visible[:actual_samples])
        target_ir_norm = normalize_tensor(target_ir[:actual_samples])
        generated_ir_norm = normalize_tensor(generated_ir[:actual_samples])
        
        # Create grid comparison plot
        fig = plt.figure(figsize=(15, 5 * actual_samples))
        gs = gridspec.GridSpec(actual_samples, 3, figure=fig)
        
        for i in range(actual_samples):
            # Visible image
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(visible_norm[i].permute(1, 2, 0).cpu().numpy())
            ax1.set_title(f'Visible Input {i+1}')
            ax1.axis('off')
            
            # Target infrared
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(target_ir_norm[i].permute(1, 2, 0).cpu().numpy())
            ax2.set_title(f'Target Infrared {i+1}')
            ax2.axis('off')
            
            # Generated infrared
            ax3 = fig.add_subplot(gs[i, 2])
            ax3.imshow(generated_ir_norm[i].permute(1, 2, 0).cpu().numpy())
            ax3.set_title(f'Generated Infrared {i+1}')
            ax3.axis('off')
        
        plt.tight_layout()
        grid_plot_path = os.path.join(save_dir, f"{filename_prefix}_grid.png")
        plt.savefig(grid_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_plots.append(grid_plot_path)
        
        # Create difference maps
        fig, axes = plt.subplots(2, actual_samples, figsize=(3 * actual_samples, 6))
        if actual_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(actual_samples):
            # Pixel-wise difference
            pixel_diff = torch.abs(target_ir_norm[i] - generated_ir_norm[i])
            pixel_diff_gray = torch.mean(pixel_diff, dim=0)
            
            axes[0, i].imshow(pixel_diff_gray.cpu().numpy(), cmap='hot')
            axes[0, i].set_title(f'Pixel Difference {i+1}')
            axes[0, i].axis('off')
            
            # Gradient difference
            target_grad = torch.abs(torch.gradient(torch.mean(target_ir_norm[i], dim=0))[0])
            generated_grad = torch.abs(torch.gradient(torch.mean(generated_ir_norm[i], dim=0))[0])
            grad_diff = torch.abs(target_grad - generated_grad)
            
            axes[1, i].imshow(grad_diff.cpu().numpy(), cmap='hot')
            axes[1, i].set_title(f'Gradient Difference {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        diff_plot_path = os.path.join(save_dir, f"{filename_prefix}_differences.png")
        plt.savefig(diff_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        saved_plots.append(diff_plot_path)
        
        return saved_plots
    
    def generate_metrics_visualization(self, metrics_history: Dict[str, List[float]], 
                                     save_dir: str, filename: str = "metrics_history.png") -> str:
        """
        Generate visualization of metrics over training/evaluation history.
        
        Args:
            metrics_history: Dictionary mapping metric names to lists of values
            save_dir: Directory to save the plot
            filename: Filename for the saved plot
            
        Returns:
            Path to the saved plot
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Set up the plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metric_names = ['fid', 'lpips', 'psnr', 'ssim', 'spectral_consistency', 'cross_spectral_correlation']
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i, (metric_name, color) in enumerate(zip(metric_names, colors)):
            if metric_name in metrics_history and len(metrics_history[metric_name]) > 0:
                axes[i].plot(metrics_history[metric_name], color=color, linewidth=2)
                axes[i].set_title(f'{metric_name.upper()}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Evaluation Step')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                if len(metrics_history[metric_name]) > 1:
                    x = np.arange(len(metrics_history[metric_name]))
                    z = np.polyfit(x, metrics_history[metric_name], 1)
                    p = np.poly1d(z)
                    axes[i].plot(x, p(x), "--", color=color, alpha=0.7)
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                           transform=axes[i].transAxes, fontsize=14)
                axes[i].set_title(f'{metric_name.upper()}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def evaluate_batch(self, visible: torch.Tensor, target_ir: torch.Tensor, 
                      generated_ir: torch.Tensor, save_dir: Optional[str] = None,
                      create_visualizations: bool = True) -> SpectralEvaluationResults:
        """
        Perform comprehensive evaluation on a batch of images.
        
        Args:
            visible: Input visible images [B, C, H, W]
            target_ir: Target infrared images [B, C, H, W]
            generated_ir: Generated infrared images [B, C, H, W]
            save_dir: Directory to save visualizations (optional)
            create_visualizations: Whether to create comparison plots
            
        Returns:
            SpectralEvaluationResults containing all metrics and visualizations
        """
        # Reset metrics
        self.reset_metrics()
        
        # Update metrics with batch
        self.update_metrics(generated_ir, target_ir, visible)
        
        # Compute all metrics
        metrics = self.compute_cross_spectral_metrics(generated_ir, target_ir, visible)
        
        # Generate visualizations if requested
        comparison_plots = []
        visual_samples = []
        
        if create_visualizations and save_dir:
            comparison_plots = self.generate_comparison_plots(
                visible, target_ir, generated_ir, save_dir
            )
            
            # Store sample images for later use
            visual_samples = [
                visible[:4].cpu(),
                target_ir[:4].cpu(),
                generated_ir[:4].cpu()
            ]
        
        return SpectralEvaluationResults(
            fid_score=metrics['fid'],
            lpips_score=metrics['lpips'],
            psnr_score=metrics['psnr'],
            ssim_score=metrics['ssim'],
            spectral_consistency=metrics['spectral_consistency'],
            cross_spectral_correlation=metrics['cross_spectral_correlation'],
            visual_samples=visual_samples,
            comparison_plots=comparison_plots
        )


def create_spectral_evaluator(device: str = 'cuda', dataset_name: str = None, 
                            data_dir: str = None) -> SpectralEvaluator:
    """
    Factory function to create a SpectralEvaluator with dataset-specific configurations.
    
    Args:
        device: Device to run computations on
        dataset_name: Name of the dataset for loading precomputed FID features
        data_dir: Data directory containing FID statistics
        
    Returns:
        Configured SpectralEvaluator instance
    """
    fid_features_path = None
    
    # Look for precomputed FID features
    if dataset_name and data_dir:
        potential_paths = [
            os.path.join(data_dir, dataset_name.lower(), 'fid_stats.pt'),
            os.path.join(data_dir, 'fid_stats.pt'),
            os.path.join(data_dir, f'{dataset_name}_fid_stats.pt')
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                fid_features_path = path
                break
    
    return SpectralEvaluator(device=device, fid_features_path=fid_features_path)