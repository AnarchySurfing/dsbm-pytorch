"""
Spectral DSBM Trainer for visible-infrared image translation.

This module extends the IPF_DBDSB trainer to support paired visible-infrared
image translation using Schrödinger Bridge methods.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from typing import Optional, Tuple, Dict, Any

from .trainer_dbdsb import IPF_DBDSB
from .data.spectral import VisibleInfraredDataset
from .data.preprocessing import SpectralImagePreprocessor, SpectralAugmentationPipeline


class SpectralDBDSBTrainer(IPF_DBDSB):
    """
    Spectral DSBM Trainer for visible-infrared image translation.
    
    Extends IPF_DBDSB to handle paired visible-infrared datasets and implements
    spectral-aware loss computation methods for cross-spectral image translation.
    """
    
    def __init__(self, 
                 visible_ds, 
                 infrared_ds, 
                 spectral_preprocessor: SpectralImagePreprocessor,
                 args, 
                 accelerator=None,
                 augmentation_pipeline: Optional[SpectralAugmentationPipeline] = None,
                 valid_visible_ds=None,
                 valid_infrared_ds=None,
                 test_visible_ds=None,
                 test_infrared_ds=None,
                 **kwargs):
        """
        Initialize the Spectral DSBM Trainer.
        
        Args:
            visible_ds: Visible image dataset (source domain)
            infrared_ds: Infrared image dataset (target domain)  
            spectral_preprocessor: Preprocessor for spectral image pairs
            args: Training arguments and configuration
            accelerator: Accelerate object for distributed training
            augmentation_pipeline: Optional augmentation pipeline for training
            valid_visible_ds: Validation visible dataset
            valid_infrared_ds: Validation infrared dataset
            test_visible_ds: Test visible dataset
            test_infrared_ds: Test infrared dataset
        """
        self.visible_ds = visible_ds
        self.infrared_ds = infrared_ds
        self.spectral_preprocessor = spectral_preprocessor
        self.augmentation_pipeline = augmentation_pipeline
        
        # Store validation and test datasets
        self.valid_visible_ds = valid_visible_ds
        self.valid_infrared_ds = valid_infrared_ds
        self.test_visible_ds = test_visible_ds
        self.test_infrared_ds = test_infrared_ds
        
        # Spectral-specific parameters
        self.spectral_loss_weight = getattr(args, 'spectral_loss_weight', 1.0)
        self.perceptual_loss_weight = getattr(args, 'perceptual_loss_weight', 0.1)
        self.consistency_loss_weight = getattr(args, 'consistency_loss_weight', 0.05)
        self.use_spectral_augmentation = getattr(args, 'use_spectral_augmentation', True)
        
        # Compute dataset statistics for normalization
        self.visible_stats = self._compute_dataset_stats(visible_ds, 'visible')
        self.infrared_stats = self._compute_dataset_stats(infrared_ds, 'infrared')
        
        # Create paired dataset for training
        paired_dataset = self._create_paired_dataset(visible_ds, infrared_ds)
        
        # Compute mean and variance for final distribution (infrared)
        mean_final, var_final = self._compute_final_distribution_stats(infrared_ds)
        
        # Create validation and test paired datasets if provided
        valid_ds = None
        if valid_visible_ds is not None and valid_infrared_ds is not None:
            valid_ds = self._create_paired_dataset(valid_visible_ds, valid_infrared_ds)
            
        test_ds = None
        if test_visible_ds is not None and test_infrared_ds is not None:
            test_ds = self._create_paired_dataset(test_visible_ds, test_infrared_ds)
        
        # Initialize parent class with paired dataset
        super().__init__(
            init_ds=paired_dataset,  # Visible images as initial distribution
            final_ds=None,  # We'll generate infrared from visible
            mean_final=mean_final,
            var_final=var_final,
            args=args,
            accelerator=accelerator,
            valid_ds=valid_ds,
            test_ds=test_ds,
            **kwargs
        )
        
        # Initialize perceptual loss network if needed
        if self.perceptual_loss_weight > 0:
            self._init_perceptual_loss()
    
    def _compute_dataset_stats(self, dataset, domain_name: str) -> Dict[str, torch.Tensor]:
        """Compute mean and std statistics for a dataset."""
        print(f"Computing {domain_name} dataset statistics...")
        
        # Sample a subset for statistics computation
        num_samples = min(1000, len(dataset))
        indices = torch.randperm(len(dataset))[:num_samples]
        
        channel_sum = torch.zeros(3)
        channel_sq_sum = torch.zeros(3)
        
        for idx in indices:
            if isinstance(dataset, VisibleInfraredDataset):
                # For paired dataset, get the appropriate domain
                visible, infrared = dataset[idx]
                image = visible if domain_name == 'visible' else infrared
            else:
                # For single domain dataset
                image = dataset[idx][0] if isinstance(dataset[idx], (tuple, list)) else dataset[idx]
            
            # Ensure image is in [-1, 1] range
            if image.max() > 1.0:
                image = image / 255.0 * 2.0 - 1.0
            
            channel_sum += image.mean(dim=[1, 2])
            channel_sq_sum += (image ** 2).mean(dim=[1, 2])
        
        mean = channel_sum / num_samples
        std = torch.sqrt(channel_sq_sum / num_samples - mean ** 2)
        
        print(f"{domain_name.capitalize()} stats - Mean: {mean.tolist()}, Std: {std.tolist()}")
        
        return {'mean': mean, 'std': std}
    
    def _create_paired_dataset(self, visible_ds, infrared_ds):
        """Create a paired dataset from visible and infrared datasets."""
        class PairedSpectralDataset(torch.utils.data.Dataset):
            def __init__(self, visible_ds, infrared_ds, preprocessor, augmentation=None):
                self.visible_ds = visible_ds
                self.infrared_ds = infrared_ds
                self.preprocessor = preprocessor
                self.augmentation = augmentation
                
                # Ensure both datasets have the same length
                assert len(visible_ds) == len(infrared_ds), \
                    f"Dataset length mismatch: visible {len(visible_ds)} vs infrared {len(infrared_ds)}"
            
            def __len__(self):
                return len(self.visible_ds)
            
            def __getitem__(self, idx):
                # Get visible and infrared images
                if isinstance(self.visible_ds, VisibleInfraredDataset):
                    visible, infrared = self.visible_ds[idx]
                else:
                    visible = self.visible_ds[idx][0] if isinstance(self.visible_ds[idx], (tuple, list)) else self.visible_ds[idx]
                    infrared = self.infrared_ds[idx][0] if isinstance(self.infrared_ds[idx], (tuple, list)) else self.infrared_ds[idx]
                
                # Apply preprocessing
                visible, infrared = self.preprocessor.preprocess_pair(visible, infrared)
                
                # Apply augmentation during training
                if self.augmentation is not None:
                    visible, infrared = self.augmentation(visible, infrared)
                
                # Return as (source, target) pair
                return visible, infrared
        
        return PairedSpectralDataset(
            visible_ds, 
            infrared_ds, 
            self.spectral_preprocessor,
            self.augmentation_pipeline if self.use_spectral_augmentation else None
        )
    
    def _compute_final_distribution_stats(self, infrared_ds) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance for the final (infrared) distribution."""
        stats = self.infrared_stats
        mean_final = stats['mean'].unsqueeze(-1).unsqueeze(-1)  # Shape: (3, 1, 1)
        var_final = (stats['std'] ** 2).unsqueeze(-1).unsqueeze(-1)  # Shape: (3, 1, 1)
        
        return mean_final, var_final
    
    def _init_perceptual_loss(self):
        """Initialize perceptual loss network (VGG features)."""
        try:
            import torchvision.models as models
            
            # Use VGG16 features for perceptual loss
            vgg = models.vgg16(pretrained=True).features
            self.perceptual_net = torch.nn.Sequential(*list(vgg.children())[:16]).eval()
            
            # Freeze parameters
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
            
            self.perceptual_net = self.perceptual_net.to(self.device)
            print("Initialized VGG16 perceptual loss network")
            
        except ImportError:
            print("Warning: torchvision not available, disabling perceptual loss")
            self.perceptual_loss_weight = 0.0
            self.perceptual_net = None
    
    def sample_batch(self, init_dl, final_dl):
        """
        Override parent method to handle spectral data sampling.
        
        Returns visible images as init_batch_x and infrared as final_batch_x.
        """
        # Sample a batch from the paired dataset
        batch = next(init_dl)
        visible_batch = batch[0]  # Source (visible)
        infrared_batch = batch[1]  # Target (infrared)
        
        # For spectral translation, we don't use conditional labels
        init_batch_y = None
        
        # Use infrared statistics for final distribution
        mean_final = self.mean_final.to(visible_batch.device)
        var_final = self.var_final.to(visible_batch.device)
        
        return visible_batch, init_batch_y, infrared_batch, mean_final, var_final
    
    def compute_spectral_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                            visible: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute spectral-aware loss components.
        
        Args:
            pred: Predicted output from the network
            target: Ground truth target
            visible: Original visible image
            t: Time step
            
        Returns:
            Dictionary containing different loss components
        """
        losses = {}
        
        # Base MSE loss
        mse_loss = F.mse_loss(pred, target)
        losses['mse'] = mse_loss
        
        # Perceptual loss (if enabled)
        if self.perceptual_loss_weight > 0 and self.perceptual_net is not None:
            # Convert from [-1, 1] to [0, 1] for VGG
            pred_norm = (pred + 1.0) / 2.0
            target_norm = (target + 1.0) / 2.0
            
            # Ensure 3 channels for VGG
            if pred_norm.shape[1] == 1:
                pred_norm = pred_norm.repeat(1, 3, 1, 1)
                target_norm = target_norm.repeat(1, 3, 1, 1)
            
            try:
                pred_features = self.perceptual_net(pred_norm)
                target_features = self.perceptual_net(target_norm)
                perceptual_loss = F.mse_loss(pred_features, target_features)
                losses['perceptual'] = perceptual_loss
            except Exception as e:
                print(f"Warning: Perceptual loss computation failed: {e}")
                losses['perceptual'] = torch.tensor(0.0, device=pred.device)
        else:
            losses['perceptual'] = torch.tensor(0.0, device=pred.device)
        
        # Spectral consistency loss (encourage spectral relationship preservation)
        if self.consistency_loss_weight > 0:
            # Compute spectral consistency between visible and predicted infrared
            visible_mean = visible.mean(dim=[2, 3], keepdim=True)
            pred_mean = pred.mean(dim=[2, 3], keepdim=True)
            
            # Encourage consistent intensity relationships
            consistency_loss = F.l1_loss(pred_mean, visible_mean * 0.8)  # IR typically darker
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0, device=pred.device)
        
        return losses
    
    def apply_net(self, x, y, t, net, fb, return_scale=False):
        """
        Override parent method to apply spectral-aware network prediction.
        
        This method handles the network forward pass and applies spectral-specific
        processing if needed.
        """
        # Get base prediction from parent method
        if return_scale:
            pred, std = super().apply_net(x, y, t, net, fb, return_scale=True)
        else:
            pred = super().apply_net(x, y, t, net, fb, return_scale=False)
            std = None
        
        # Apply spectral-specific post-processing if needed
        # For now, we use the base prediction as-is
        # Future enhancements could include spectral-aware transformations
        
        if return_scale:
            return pred, std
        else:
            return pred
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                    visible: torch.Tensor, t: torch.Tensor, 
                    forward_or_backward: str, std: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral-aware training loss.
        
        Args:
            pred: Network prediction
            target: Ground truth target
            visible: Original visible image (for consistency loss)
            t: Time step
            forward_or_backward: Training direction
            std: Standard deviation for loss scaling
            
        Returns:
            Combined loss tensor
        """
        # Apply loss scaling if enabled
        if self.args.loss_scale:
            loss_scale = std
        else:
            loss_scale = 1.0
        
        # Base MSE loss with scaling
        mse_loss = F.mse_loss(loss_scale * pred, loss_scale * target)
        total_loss = self.spectral_loss_weight * mse_loss
        
        # Perceptual loss (if enabled)
        if self.perceptual_loss_weight > 0 and self.perceptual_net is not None:
            try:
                # Convert from [-1, 1] to [0, 1] for VGG
                pred_norm = (pred + 1.0) / 2.0
                target_norm = (target + 1.0) / 2.0
                
                # Ensure 3 channels for VGG
                if pred_norm.shape[1] == 1:
                    pred_norm = pred_norm.repeat(1, 3, 1, 1)
                    target_norm = target_norm.repeat(1, 3, 1, 1)
                
                pred_features = self.perceptual_net(pred_norm)
                target_features = self.perceptual_net(target_norm)
                perceptual_loss = F.mse_loss(pred_features, target_features)
                total_loss += self.perceptual_loss_weight * perceptual_loss
                
            except Exception as e:
                print(f"Warning: Perceptual loss computation failed: {e}")
                perceptual_loss = torch.tensor(0.0, device=pred.device)
        else:
            perceptual_loss = torch.tensor(0.0, device=pred.device)
        
        # Spectral consistency loss
        if self.consistency_loss_weight > 0:
            # Compute spectral consistency between visible and predicted infrared
            visible_mean = visible.mean(dim=[2, 3], keepdim=True)
            pred_mean = pred.mean(dim=[2, 3], keepdim=True)
            
            # Encourage consistent intensity relationships (IR typically darker)
            consistency_loss = F.l1_loss(pred_mean, visible_mean * 0.8)
            total_loss += self.consistency_loss_weight * consistency_loss
        else:
            consistency_loss = torch.tensor(0.0, device=pred.device)
        
        # Log individual loss components
        if hasattr(self, 'logger') and self.logger is not None:
            try:
                self.logger.log_metrics({
                    f'{forward_or_backward}_mse_loss': mse_loss.item(),
                    f'{forward_or_backward}_perceptual_loss': perceptual_loss.item(),
                    f'{forward_or_backward}_consistency_loss': consistency_loss.item(),
                    f'{forward_or_backward}_total_loss': total_loss.item(),
                })
            except:
                pass  # Ignore logging errors during training
        
        return total_loss
    
    def ipf_iter(self, forward_or_backward, n):
        """
        Override parent method to implement spectral-aware training iteration.
        
        This method extends the base IPF iteration with spectral-specific
        loss computation and logging.
        """
        if self.first_pass:
            step = self.step
        else:
            step = 1
        
        self.set_seed(seed=self.compute_current_step(step - 1, n) * self.accelerator.num_processes + self.accelerator.process_index)
        self.i, self.n, self.fb = step - 1, n, forward_or_backward

        if (not self.first_pass) and (not self.args.use_prev_net):
            self.build_models(forward_or_backward)
            self.build_optimizers(forward_or_backward)

        self.accelerate(forward_or_backward)

        if (forward_or_backward not in self.ema_helpers.keys()) or ((not self.first_pass) and (not self.args.use_prev_net)):
            self.update_ema(forward_or_backward)
        
        num_iter = self.compute_max_iter(forward_or_backward, n)
        
        def first_it_fn(forward_or_backward, n):
            if self.args.first_coupling == 'ref':
                first_it = ((n == 1) and (forward_or_backward == 'b'))
            elif self.args.first_coupling == 'ind':
                first_it = (n == 1)
            return first_it
        first_it = first_it_fn(forward_or_backward, n)

        # Import tqdm for progress bar
        from tqdm import tqdm

        # Training loop with spectral-aware loss computation
        for i in tqdm(range(step, num_iter + 1), mininterval=30):
            
            # Refresh cache data loader
            if (i == step) or ((i-1) % self.args.cache_refresh_stride == 0):
                new_dl = None
                torch.cuda.empty_cache()
                if not first_it:
                    new_dl = self.new_cacheloader(*self.compute_prev_it(forward_or_backward, n), refresh_idx=(i-1) // self.args.cache_refresh_stride)

            self.net[forward_or_backward].train()

            self.set_seed(seed=self.compute_current_step(i, n) * self.accelerator.num_processes + self.accelerator.process_index)

            y = None
            # Sample training data
            if first_it:
                x0, y, x1, _, _ = self.sample_batch(self.init_dl, self.final_dl)
            else:
                if self.cdsb:
                    x0, x1, y = next(new_dl)
                else:
                    x0, x1 = next(new_dl)

            x0, x1 = x0.to(self.device), x1.to(self.device)
            x0, x1 = x0.repeat_interleave(self.num_repeat_data, dim=0), x1.repeat_interleave(self.num_repeat_data, dim=0)
            
            # Store original visible images for consistency loss
            visible_original = x0.clone()
            
            x, t, out = self.langevin.get_train_tuple(x0, x1, fb=forward_or_backward, first_it=first_it)

            if self.cdsb:
                y = y.to(self.device)
                y = y.repeat_interleave(self.num_repeat_data, dim=0)

            # Compute prediction and loss with spectral awareness
            pred, std = self.apply_net(x, y, t, net=self.net[forward_or_backward], fb=forward_or_backward, return_scale=True)

            # Use spectral-aware loss computation
            loss = self.compute_loss(pred, out, visible_original, t, forward_or_backward, std)

            self.accelerator.backward(loss)

            if self.grad_clipping:
                clipping_param = self.args.grad_clip
                total_norm = self.accelerator.clip_grad_norm_(self.net[forward_or_backward].parameters(), clipping_param)
            else:
                total_norm = 0.

            # Log metrics
            if i == 1 or i % self.stride_log == 0 or i == num_iter:
                self.logger.log_metrics({'fb': forward_or_backward,
                                         'ipf': n,
                                         'loss': loss,
                                         'grad_norm': total_norm,
                                         "cache_epochs": self.cache_epochs,
                                         "num_repeat_data": self.num_repeat_data,
                                         "data_epochs": self.data_epochs}, step=self.compute_current_step(i, n))

            # Update model parameters
            self.optimizer[forward_or_backward].step()
            self.optimizer[forward_or_backward].zero_grad(set_to_none=True)
            if self.args.ema:
                self.ema_helpers[forward_or_backward].update(self.accelerator.unwrap_model(self.net[forward_or_backward]))

            self.i, self.n, self.fb = i, n, forward_or_backward

            if i != num_iter:
                self.save_step(i, n, forward_or_backward)

        # Pre-cache current iter at end of training
        new_dl = None
        self.save_ckpt(num_iter, n, forward_or_backward)
        if not first_it_fn(*self.compute_next_it(forward_or_backward, n)):
            self.new_cacheloader(forward_or_backward, n, build_dataloader=False)

        self.save_step(num_iter, n, forward_or_backward)

        self.net[forward_or_backward] = self.accelerator.unwrap_model(self.net[forward_or_backward])
        self.clear()
        self.first_pass = False

    def generate_infrared(self, visible_images: torch.Tensor, 
                         num_steps: Optional[int] = None,
                         use_ode: bool = False) -> torch.Tensor:
        """
        Generate infrared images from visible images.
        
        Args:
            visible_images: Batch of visible images with shape (B, C, H, W)
            num_steps: Number of sampling steps (uses default if None)
            use_ode: Whether to use ODE sampling for faster generation
            
        Returns:
            Generated infrared images with same shape as input
        """
        self.net['f'].eval()  # Use forward network for visible->infrared
        
        with torch.no_grad():
            visible_images = visible_images.to(self.device)
            
            # Preprocess visible images if needed
            if hasattr(self, 'spectral_preprocessor'):
                # Assume images are already preprocessed if coming from dataset
                pass
            
            if use_ode:
                # Use ODE sampling for faster generation
                try:
                    infrared_seq, nfe = self.forward_sample_ode(
                        visible_images, 
                        init_batch_y=None, 
                        permute=True
                    )
                    infrared_generated = infrared_seq[-1]  # Take final step
                    print(f"ODE sampling completed with {nfe} function evaluations")
                except:
                    # Fallback to SDE sampling
                    print("ODE sampling failed, falling back to SDE sampling")
                    infrared_seq, steps = self.forward_sample(
                        visible_images, 
                        init_batch_y=None, 
                        permute=True,
                        num_steps=num_steps
                    )
                    infrared_generated = infrared_seq[-1]
            else:
                # Use standard SDE sampling
                infrared_seq, steps = self.forward_sample(
                    visible_images, 
                    init_batch_y=None, 
                    permute=True,
                    num_steps=num_steps
                )
                infrared_generated = infrared_seq[-1]  # Take final step
            
            # Post-process generated images if needed
            if hasattr(self, 'spectral_preprocessor'):
                # Convert back to uint8 range for visualization/saving
                _, infrared_uint8 = self.spectral_preprocessor.postprocess_pair(
                    visible_images, infrared_generated
                )
                return infrared_uint8.float() / 255.0 * 2.0 - 1.0  # Back to [-1, 1]
            
            return infrared_generated
    
    def evaluate_spectral_metrics(self, visible_batch: torch.Tensor, 
                                 infrared_batch: torch.Tensor,
                                 generated_batch: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate spectral-specific metrics.
        
        Args:
            visible_batch: Original visible images
            infrared_batch: Ground truth infrared images  
            generated_batch: Generated infrared images
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        with torch.no_grad():
            # MSE between generated and ground truth
            mse = F.mse_loss(generated_batch, infrared_batch)
            metrics['mse'] = mse.item()
            
            # PSNR
            psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))  # Range is [-1, 1]
            metrics['psnr'] = psnr.item()
            
            # L1 loss
            l1_loss = F.l1_loss(generated_batch, infrared_batch)
            metrics['l1'] = l1_loss.item()
            
            # Spectral consistency (correlation between visible and generated)
            visible_flat = visible_batch.view(visible_batch.shape[0], -1)
            generated_flat = generated_batch.view(generated_batch.shape[0], -1)
            
            # Compute correlation coefficient
            visible_centered = visible_flat - visible_flat.mean(dim=1, keepdim=True)
            generated_centered = generated_flat - generated_flat.mean(dim=1, keepdim=True)
            
            correlation = (visible_centered * generated_centered).sum(dim=1) / (
                torch.sqrt((visible_centered ** 2).sum(dim=1)) * 
                torch.sqrt((generated_centered ** 2).sum(dim=1)) + 1e-8
            )
            metrics['spectral_correlation'] = correlation.mean().item()
        
        return metrics
    
    def save_spectral_samples(self, visible_batch: torch.Tensor,
                            infrared_batch: torch.Tensor,
                            generated_batch: torch.Tensor,
                            save_path: str,
                            num_samples: int = 8):
        """
        Save spectral image samples for visualization.
        
        Args:
            visible_batch: Visible images
            infrared_batch: Ground truth infrared images
            generated_batch: Generated infrared images
            save_path: Path to save the visualization
            num_samples: Number of samples to save
        """
        import torchvision.utils as vutils
        
        num_samples = min(num_samples, visible_batch.shape[0])
        
        # Select samples
        visible_samples = visible_batch[:num_samples]
        infrared_samples = infrared_batch[:num_samples]
        generated_samples = generated_batch[:num_samples]
        
        # Convert to [0, 1] range for visualization
        visible_vis = (visible_samples + 1.0) / 2.0
        infrared_vis = (infrared_samples + 1.0) / 2.0
        generated_vis = (generated_samples + 1.0) / 2.0
        
        # Create comparison grid: [visible, ground_truth_ir, generated_ir]
        comparison = torch.cat([visible_vis, infrared_vis, generated_vis], dim=0)
        
        # Save grid
        vutils.save_image(
            comparison,
            save_path,
            nrow=num_samples,
            normalize=False,
            padding=2
        )
        
        print(f"Saved spectral samples to {save_path}")
    
    def get_logger(self, name='spectral_logs'):
        """Override to use spectral-specific logger name."""
        return super().get_logger(name)