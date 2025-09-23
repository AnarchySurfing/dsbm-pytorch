"""
Image preprocessing module for spectral image translation.

This module provides preprocessing and postprocessing functions specifically
designed for paired visible-infrared image datasets used in Schrödinger Bridge
based image translation.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import Tuple, Union, Optional, List
import numpy as np
from PIL import Image
import warnings


class SpectralImagePreprocessor:
    """
    Preprocessor for paired visible-infrared images.
    
    Handles normalization, resizing, augmentation, and other preprocessing
    operations for spectral image pairs while maintaining consistency
    between visible and infrared images.
    """
    
    def __init__(self, 
                 target_size: Optional[Union[int, Tuple[int, int]]] = None,
                 normalize_range: Tuple[float, float] = (-1.0, 1.0),
                 maintain_aspect_ratio: bool = True,
                 interpolation_mode: str = 'bilinear'):
        """
        Initialize the spectral image preprocessor.
        
        Args:
            target_size: Target size for images. If int, creates square images.
                        If tuple, uses (height, width). If None, no resizing.
            normalize_range: Target normalization range (min, max)
            maintain_aspect_ratio: Whether to maintain aspect ratio during resize
            interpolation_mode: Interpolation mode for resizing ('bilinear', 'nearest', etc.)
        """
        self.target_size = target_size
        self.normalize_range = normalize_range
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.interpolation_mode = interpolation_mode
        
        # Validate normalize_range
        if normalize_range[0] >= normalize_range[1]:
            raise ValueError("normalize_range[0] must be less than normalize_range[1]")
    
    def preprocess_pair(self, 
                       visible: torch.Tensor, 
                       infrared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess a visible-infrared image pair.
        
        Args:
            visible: Visible image tensor with shape (C, H, W) or (H, W, C)
            infrared: Infrared image tensor with shape (C, H, W) or (H, W, C)
            
        Returns:
            Tuple of preprocessed (visible, infrared) tensors
        """
        # Ensure tensors are in (C, H, W) format
        visible = self._ensure_chw_format(visible)
        infrared = self._ensure_chw_format(infrared)
        
        # Validate input shapes
        if visible.shape != infrared.shape:
            raise ValueError(f"Visible and infrared shapes must match: {visible.shape} vs {infrared.shape}")
        
        # Resize if target size is specified
        if self.target_size is not None:
            visible, infrared = self._resize_pair(visible, infrared)
        
        # Normalize to target range
        visible = self._normalize_tensor(visible)
        infrared = self._normalize_tensor(infrared)
        
        return visible, infrared
    
    def postprocess_pair(self, 
                        visible: torch.Tensor, 
                        infrared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Postprocess a visible-infrared image pair back to [0, 255] uint8 range.
        
        Args:
            visible: Preprocessed visible image tensor
            infrared: Preprocessed infrared image tensor
            
        Returns:
            Tuple of postprocessed (visible, infrared) tensors in [0, 255] range
        """
        visible = self._denormalize_tensor(visible)
        infrared = self._denormalize_tensor(infrared)
        
        return visible, infrared
    
    def _ensure_chw_format(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is in (C, H, W) format."""
        if tensor.dim() == 3:
            if tensor.shape[0] in [1, 3]:  # Already (C, H, W)
                return tensor
            elif tensor.shape[2] in [1, 3]:  # (H, W, C) format
                return tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Cannot determine format of tensor with shape {tensor.shape}")
        else:
            raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D")
    
    def _resize_pair(self, 
                    visible: torch.Tensor, 
                    infrared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize image pair to target size."""
        if isinstance(self.target_size, int):
            target_h = target_w = self.target_size
        else:
            target_h, target_w = self.target_size
        
        # Get interpolation mode
        if self.interpolation_mode == 'bilinear':
            mode = 'bilinear'
            align_corners = False
        elif self.interpolation_mode == 'nearest':
            mode = 'nearest'
            align_corners = None
        else:
            mode = self.interpolation_mode
            align_corners = False
        
        # Resize both images with same parameters
        visible = F.interpolate(
            visible.unsqueeze(0), 
            size=(target_h, target_w), 
            mode=mode,
            align_corners=align_corners
        ).squeeze(0)
        
        infrared = F.interpolate(
            infrared.unsqueeze(0), 
            size=(target_h, target_w), 
            mode=mode,
            align_corners=align_corners
        ).squeeze(0)
        
        return visible, infrared
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to target range."""
        # Assume input is in [0, 255] or [0, 1] range
        if tensor.max() > 1.0:
            # Input is in [0, 255] range
            tensor = tensor / 255.0
        
        # Now tensor is in [0, 1] range, convert to target range
        min_val, max_val = self.normalize_range
        tensor = tensor * (max_val - min_val) + min_val
        
        return tensor.clamp(min_val, max_val)
    
    def _denormalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor back to [0, 255] uint8 range."""
        min_val, max_val = self.normalize_range
        
        # Convert from target range to [0, 1]
        tensor = (tensor - min_val) / (max_val - min_val)
        tensor = tensor.clamp(0, 1)
        
        # Convert to [0, 255] uint8
        tensor = (tensor * 255).round().clamp(0, 255).to(torch.uint8)
        
        return tensor


class SpectralAugmentationPipeline:
    """
    Augmentation pipeline for paired visible-infrared images.
    
    Applies consistent augmentations to both visible and infrared images
    to maintain spatial correspondence.
    """
    
    def __init__(self, 
                 horizontal_flip_prob: float = 0.5,
                 vertical_flip_prob: float = 0.0,
                 rotation_degrees: Union[float, Tuple[float, float]] = 0,
                 brightness_factor: float = 0.1,
                 contrast_factor: float = 0.1,
                 saturation_factor: float = 0.1,
                 hue_factor: float = 0.05,
                 apply_color_jitter_to_infrared: bool = False):
        """
        Initialize the augmentation pipeline.
        
        Args:
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            rotation_degrees: Range of rotation degrees. If float, uses (-degrees, degrees)
            brightness_factor: Brightness jitter factor for visible images
            contrast_factor: Contrast jitter factor for visible images
            saturation_factor: Saturation jitter factor for visible images
            hue_factor: Hue jitter factor for visible images
            apply_color_jitter_to_infrared: Whether to apply color jitter to infrared images
        """
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_degrees = rotation_degrees
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.hue_factor = hue_factor
        self.apply_color_jitter_to_infrared = apply_color_jitter_to_infrared
        
        # Create color jitter transform
        if any([brightness_factor, contrast_factor, saturation_factor, hue_factor]):
            self.color_jitter = transforms.ColorJitter(
                brightness=brightness_factor,
                contrast=contrast_factor,
                saturation=saturation_factor,
                hue=hue_factor
            )
        else:
            self.color_jitter = None
    
    def __call__(self, 
                 visible: torch.Tensor, 
                 infrared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to visible-infrared pair.
        
        Args:
            visible: Visible image tensor (C, H, W)
            infrared: Infrared image tensor (C, H, W)
            
        Returns:
            Tuple of augmented (visible, infrared) tensors
        """
        # Apply geometric transformations consistently
        visible, infrared = self._apply_geometric_transforms(visible, infrared)
        
        # Apply color transformations
        visible = self._apply_color_transforms(visible, is_infrared=False)
        if self.apply_color_jitter_to_infrared:
            infrared = self._apply_color_transforms(infrared, is_infrared=True)
        
        return visible, infrared
    
    def _apply_geometric_transforms(self, 
                                  visible: torch.Tensor, 
                                  infrared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply consistent geometric transformations to both images."""
        # Random horizontal flip
        if torch.rand(1) < self.horizontal_flip_prob:
            visible = TF.hflip(visible)
            infrared = TF.hflip(infrared)
        
        # Random vertical flip
        if torch.rand(1) < self.vertical_flip_prob:
            visible = TF.vflip(visible)
            infrared = TF.vflip(infrared)
        
        # Random rotation
        if self.rotation_degrees != 0:
            if isinstance(self.rotation_degrees, (tuple, list)):
                angle_range = self.rotation_degrees[1] - self.rotation_degrees[0]
                angle = torch.rand(1) * angle_range + self.rotation_degrees[0]
            else:
                angle = torch.rand(1) * 2 * self.rotation_degrees - self.rotation_degrees
            
            visible = TF.rotate(visible, angle.item())
            infrared = TF.rotate(infrared, angle.item())
        
        return visible, infrared
    
    def _apply_color_transforms(self, image: torch.Tensor, is_infrared: bool = False) -> torch.Tensor:
        """Apply color transformations to a single image."""
        if self.color_jitter is None:
            return image
        
        # Convert to PIL for color jitter, then back to tensor
        # Note: This assumes input is in [0, 1] range
        if image.max() <= 1.0:
            pil_image = TF.to_pil_image(image)
        else:
            # If in [-1, 1] range, convert to [0, 1] first
            normalized = (image + 1.0) / 2.0
            pil_image = TF.to_pil_image(normalized)
        
        # Apply color jitter
        augmented_pil = self.color_jitter(pil_image)
        
        # Convert back to tensor
        augmented_tensor = TF.to_tensor(augmented_pil)
        
        # Convert back to original range if needed
        if image.max() <= 1.0:
            return augmented_tensor
        else:
            return augmented_tensor * 2.0 - 1.0


# Convenience functions for common preprocessing operations
def normalize_rgb_to_range(tensor: torch.Tensor, 
                          target_range: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """
    Normalize RGB tensor to specified range.
    
    Args:
        tensor: Input tensor, assumed to be in [0, 255] or [0, 1] range
        target_range: Target normalization range (min, max)
        
    Returns:
        Normalized tensor in target range
    """
    # Detect input range
    if tensor.max() > 1.0:
        # Input is in [0, 255] range
        tensor = tensor / 255.0
    
    # Convert from [0, 1] to target range
    min_val, max_val = target_range
    tensor = tensor * (max_val - min_val) + min_val
    
    return tensor.clamp(min_val, max_val)


def denormalize_to_uint8(tensor: torch.Tensor, 
                        source_range: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """
    Denormalize tensor back to [0, 255] uint8 range.
    
    Args:
        tensor: Input tensor in source_range
        source_range: Source normalization range (min, max)
        
    Returns:
        Tensor in [0, 255] uint8 range
    """
    min_val, max_val = source_range
    
    # Convert from source range to [0, 1]
    tensor = (tensor - min_val) / (max_val - min_val)
    tensor = tensor.clamp(0, 1)
    
    # Convert to [0, 255] uint8
    tensor = (tensor * 255).round().clamp(0, 255).to(torch.uint8)
    
    return tensor


def resize_image_pair(visible: torch.Tensor, 
                     infrared: torch.Tensor,
                     target_size: Union[int, Tuple[int, int]],
                     mode: str = 'bilinear') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Resize a visible-infrared image pair to target size.
    
    Args:
        visible: Visible image tensor (C, H, W)
        infrared: Infrared image tensor (C, H, W)
        target_size: Target size. If int, creates square images.
        mode: Interpolation mode ('bilinear', 'nearest', etc.)
        
    Returns:
        Tuple of resized (visible, infrared) tensors
    """
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size
    
    # Set interpolation parameters
    if mode == 'bilinear':
        align_corners = False
    elif mode == 'nearest':
        align_corners = None
    else:
        align_corners = False
    
    # Resize both images
    visible = F.interpolate(
        visible.unsqueeze(0), 
        size=(target_h, target_w), 
        mode=mode,
        align_corners=align_corners
    ).squeeze(0)
    
    infrared = F.interpolate(
        infrared.unsqueeze(0), 
        size=(target_h, target_w), 
        mode=mode,
        align_corners=align_corners
    ).squeeze(0)
    
    return visible, infrared


def compute_image_statistics(dataset, 
                           num_samples: Optional[int] = None,
                           channels: int = 3) -> dict:
    """
    Compute mean and standard deviation statistics for a dataset.
    
    Args:
        dataset: Dataset object with __getitem__ method returning (visible, infrared)
        num_samples: Number of samples to use. If None, uses entire dataset
        channels: Number of channels (3 for RGB)
        
    Returns:
        Dictionary with statistics for visible and infrared images
    """
    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))
    
    visible_sum = torch.zeros(channels)
    infrared_sum = torch.zeros(channels)
    visible_sq_sum = torch.zeros(channels)
    infrared_sq_sum = torch.zeros(channels)
    
    print(f"Computing statistics for {num_samples} samples...")
    
    for i in range(num_samples):
        visible, infrared = dataset[i]
        
        # Compute per-channel means
        visible_sum += visible.mean(dim=[1, 2])
        infrared_sum += infrared.mean(dim=[1, 2])
        
        # Compute per-channel squared means for std calculation
        visible_sq_sum += (visible ** 2).mean(dim=[1, 2])
        infrared_sq_sum += (infrared ** 2).mean(dim=[1, 2])
    
    # Calculate final statistics
    visible_mean = visible_sum / num_samples
    infrared_mean = infrared_sum / num_samples
    visible_std = torch.sqrt(visible_sq_sum / num_samples - visible_mean ** 2)
    infrared_std = torch.sqrt(infrared_sq_sum / num_samples - infrared_mean ** 2)
    
    return {
        'visible': {
            'mean': visible_mean.tolist(),
            'std': visible_std.tolist()
        },
        'infrared': {
            'mean': infrared_mean.tolist(),
            'std': infrared_std.tolist()
        },
        'num_samples': num_samples
    }