"""
Paired spectral dataset infrastructure for visible-infrared image translation.

This module provides base classes and concrete implementations for handling
paired visible-infrared datasets used in Schrödinger Bridge-based image translation.
"""

import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from abc import ABC, abstractmethod
import warnings


class PairedSpectralDataset(Dataset, ABC):
    """
    Abstract base class for paired visible-infrared datasets.
    
    This class defines the standard interface for paired spectral datasets
    where each sample consists of a visible RGB image and its corresponding
    infrared RGB image.
    """
    
    def __init__(self, root_dir: str, transform=None, target_transform=None):
        """
        Initialize the paired spectral dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            transform: Optional transform to apply to visible images
            target_transform: Optional transform to apply to infrared images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of paired samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a paired sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (visible_rgb, infrared_rgb) tensors with shape (C, H, W)
            Both tensors should be normalized to [-1, 1] range
        """
        pass
    
    @abstractmethod
    def get_metadata(self, index: int) -> Dict[str, Any]:
        """
        Get metadata for a specific sample.
        
        Args:
            index: Index of the sample
            
        Returns:
            Dictionary containing metadata like file paths, capture conditions, etc.
        """
        pass
    
    def validate_pair(self, visible_path: Path, infrared_path: Path) -> bool:
        """
        Validate that a visible-infrared pair is compatible.
        
        Args:
            visible_path: Path to visible image
            infrared_path: Path to infrared image
            
        Returns:
            True if the pair is valid, False otherwise
        """
        try:
            # Check if both files exist
            if not visible_path.exists() or not infrared_path.exists():
                return False
            
            # Load images to check dimensions
            visible_img = Image.open(visible_path)
            infrared_img = Image.open(infrared_path)
            
            # Check if dimensions are compatible
            if visible_img.size != infrared_img.size:
                warnings.warn(
                    f"Size mismatch: visible {visible_img.size} vs infrared {infrared_img.size} "
                    f"for pair {visible_path.name}"
                )
                return False
            
            # Check if both are RGB images
            if visible_img.mode != 'RGB' or infrared_img.mode != 'RGB':
                warnings.warn(
                    f"Mode mismatch: visible {visible_img.mode} vs infrared {infrared_img.mode} "
                    f"for pair {visible_path.name}"
                )
                return False
            
            return True
            
        except Exception as e:
            warnings.warn(f"Error validating pair {visible_path.name}: {str(e)}")
            return False
    
    def _load_image_as_tensor(self, image_path: Path) -> torch.Tensor:
        """
        Load an image and convert it to a normalized tensor.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tensor with shape (C, H, W) normalized to [-1, 1] range
        """
        try:
            pil_image = Image.open(image_path).convert('RGB')
            np_image = np.array(pil_image)  # 0 to 255 integer
            
            # Scale floats between -1 and 1
            tensor_image = (torch.tensor(np_image, dtype=torch.float32) / 255.0) * 2 - 1
            
            # Transpose from (H, W, C) to (C, H, W)
            tensor_image = tensor_image.permute(2, 0, 1)
            
            return tensor_image
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")


class VisibleInfraredDataset(PairedSpectralDataset):
    """
    Concrete implementation for directory-based visible-infrared paired datasets.
    
    Expected directory structure:
    root_dir/
    ├── visible/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── infrared/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    
    Images with the same filename are considered pairs.
    """
    
    def __init__(self, 
                 root_dir: str,
                 visible_subdir: str = "visible",
                 infrared_subdir: str = "infrared",
                 image_size: Optional[int] = None,
                 transform=None,
                 target_transform=None,
                 validate_pairs: bool = True):
        """
        Initialize the visible-infrared dataset.
        
        Args:
            root_dir: Root directory containing visible and infrared subdirectories
            visible_subdir: Name of subdirectory containing visible images
            infrared_subdir: Name of subdirectory containing infrared images
            image_size: Optional target size for images (will resize if provided)
            transform: Optional transform to apply to visible images
            target_transform: Optional transform to apply to infrared images
            validate_pairs: Whether to validate all pairs during initialization
        """
        super().__init__(root_dir, transform, target_transform)
        
        self.visible_dir = self.root_dir / visible_subdir
        self.infrared_dir = self.root_dir / infrared_subdir
        self.image_size = image_size
        
        # Validate directory structure
        if not self.visible_dir.exists():
            raise FileNotFoundError(f"Visible directory not found: {self.visible_dir}")
        if not self.infrared_dir.exists():
            raise FileNotFoundError(f"Infrared directory not found: {self.infrared_dir}")
        
        # Find all valid image pairs
        self.valid_pairs = self._find_valid_pairs(validate_pairs)
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid visible-infrared pairs found in the dataset")
        
        print(f"Found {len(self.valid_pairs)} valid visible-infrared pairs")
    
    def _find_valid_pairs(self, validate: bool = True) -> List[Tuple[Path, Path]]:
        """
        Find all valid visible-infrared pairs in the dataset.
        
        Args:
            validate: Whether to validate each pair
            
        Returns:
            List of (visible_path, infrared_path) tuples
        """
        # Get all image files from visible directory
        visible_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        visible_files = []
        
        for ext in visible_extensions:
            visible_files.extend(self.visible_dir.glob(f"*{ext}"))
            visible_files.extend(self.visible_dir.glob(f"*{ext.upper()}"))
        
        valid_pairs = []
        
        for visible_path in visible_files:
            # Look for corresponding infrared image
            infrared_path = self.infrared_dir / visible_path.name
            
            # Try different extensions if exact match not found
            if not infrared_path.exists():
                stem = visible_path.stem
                found = False
                for ext in visible_extensions:
                    candidate = self.infrared_dir / f"{stem}{ext}"
                    if candidate.exists():
                        infrared_path = candidate
                        found = True
                        break
                
                if not found:
                    continue
            
            # Validate the pair if requested
            if validate and not self.validate_pair(visible_path, infrared_path):
                continue
            
            valid_pairs.append((visible_path, infrared_path))
        
        return sorted(valid_pairs)
    
    def __len__(self) -> int:
        """Return the total number of valid pairs."""
        return len(self.valid_pairs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a visible-infrared pair from the dataset.
        
        Args:
            index: Index of the pair to retrieve
            
        Returns:
            Tuple of (visible_rgb, infrared_rgb) tensors with shape (C, H, W)
        """
        if index >= len(self.valid_pairs):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.valid_pairs)}")
        
        visible_path, infrared_path = self.valid_pairs[index]
        
        # Load images as tensors
        visible_tensor = self._load_image_as_tensor(visible_path)
        infrared_tensor = self._load_image_as_tensor(infrared_path)
        
        # Resize if requested
        if self.image_size is not None:
            import torch.nn.functional as F
            visible_tensor = F.interpolate(
                visible_tensor.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            infrared_tensor = F.interpolate(
                infrared_tensor.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Apply transforms if provided
        if self.transform is not None:
            visible_tensor = self.transform(visible_tensor)
        
        if self.target_transform is not None:
            infrared_tensor = self.target_transform(infrared_tensor)
        
        return visible_tensor, infrared_tensor
    
    def get_metadata(self, index: int) -> Dict[str, Any]:
        """
        Get metadata for a specific pair.
        
        Args:
            index: Index of the pair
            
        Returns:
            Dictionary containing file paths and other metadata
        """
        if index >= len(self.valid_pairs):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.valid_pairs)}")
        
        visible_path, infrared_path = self.valid_pairs[index]
        
        return {
            'visible_path': str(visible_path),
            'infrared_path': str(infrared_path),
            'visible_filename': visible_path.name,
            'infrared_filename': infrared_path.name,
            'pair_id': visible_path.stem,
            'index': index
        }
    
    def get_pair_paths(self, index: int) -> Tuple[Path, Path]:
        """
        Get the file paths for a specific pair.
        
        Args:
            index: Index of the pair
            
        Returns:
            Tuple of (visible_path, infrared_path)
        """
        if index >= len(self.valid_pairs):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.valid_pairs)}")
        
        return self.valid_pairs[index]