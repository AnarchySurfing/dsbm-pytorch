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
        初始化配对光谱数据集。
        
        Args:
            root_dir: 包含数据集的根目录
            transform: 可选的，应用于可见光图像的变换
            target_transform: 可选的，应用于红外图像的变换
        """
        # 将根目录字符串转换为Path对象，便于文件系统操作
        self.root_dir = Path(root_dir)
        # 存储应用于可见光图像的变换
        self.transform = transform
        # 存储应用于红外图像的变换
        self.target_transform = target_transform
        
        # 检查根目录是否存在
        if not self.root_dir.exists():
            # 如果目录不存在，则抛出文件未找到错误
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
    
    @abstractmethod
    def __len__(self) -> int:
        """返回数据集中配对样本的总数。"""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从数据集中获取一个配对样本。
        
        Args:
            index: 要检索的样本索引
            
        Returns:
            (visible_rgb, infrared_rgb) 张量的元组，形状为 (C, H, W)
            两个张量都应归一化到 [-1, 1] 范围
        """
        pass
    
    @abstractmethod
    def get_metadata(self, index: int) -> Dict[str, Any]:
        """
        获取特定样本的元数据。
        
        Args:
            index: 样本的索引
            
        Returns:
            包含文件路径、捕获条件等元数据的字典
        """
        pass
    
    def validate_pair(self, visible_path: Path, infrared_path: Path) -> bool:
        """
        验证可见光-红外图像对是否兼容。
        
        Args:
            visible_path: 可见光图像的路径
            infrared_path: 红外图像的路径
            
        Returns:
            如果图像对有效则返回 True，否则返回 False
        """
        try:
            # 检查两个文件是否存在
            if not visible_path.exists() or not infrared_path.exists():
                return False
            
            # 加载图像以检查尺寸
            visible_img = Image.open(visible_path)
            infrared_img = Image.open(infrared_path)
            
            # 检查图像尺寸是否兼容
            if visible_img.size != infrared_img.size:
                # 如果尺寸不匹配，发出警告
                warnings.warn(
                    f"Size mismatch: visible {visible_img.size} vs infrared {infrared_img.size} "
                    f"for pair {visible_path.name}"
                )
                return False
            
            # 检查两个图像是否都是RGB模式
            if visible_img.mode != 'RGB' or infrared_img.mode != 'RGB':
                # 如果模式不匹配，发出警告
                warnings.warn(
                    f"Mode mismatch: visible {visible_img.mode} vs infrared {infrared_img.mode} "
                    f"for pair {visible_path.name}"
                )
                return False
            
            # 如果所有检查都通过，则图像对有效
            return True
            
        # 捕获验证过程中可能发生的任何异常
        except Exception as e:
            # 发出警告并返回False
            warnings.warn(f"Error validating pair {visible_path.name}: {str(e)}")
            return False
    
    def _load_image_as_tensor(self, image_path: Path) -> torch.Tensor:
        """
        加载图像并将其转换为归一化张量。
        
        Args:
            image_path: 图像文件的路径
            
        Returns:
            形状为 (C, H, W) 的张量，归一化到 [-1, 1] 范围
        """
        try:
            # 使用PIL加载图像并转换为RGB模式
            pil_image = Image.open(image_path).convert('RGB')
            # 将PIL图像转换为NumPy数组，像素值范围为0到255
            np_image = np.array(pil_image)  # 0 to 255 integer
            
            # 将NumPy数组转换为PyTorch张量，并归一化到 [-1, 1] 范围
            # (像素值 / 255.0) * 2 - 1
            tensor_image = (torch.tensor(np_image, dtype=torch.float32) / 255.0) * 2 - 1
            
            # 将张量从 (H, W, C) 维度转置为 (C, H, W)
            tensor_image = tensor_image.permute(2, 0, 1)
            
            return tensor_image
            
        # 捕获加载图像过程中可能发生的任何异常
        except Exception as e:
            # 抛出运行时错误
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")


class VisibleInfraredDataset(PairedSpectralDataset):
    """
    基于目录的可见光-红外配对数据集的具体实现。
    
    预期的目录结构:
    root_dir/
    ├── visible/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── infrared/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    
    文件名相同的图像被视为一对。
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
        初始化可见光-红外数据集。
        
        Args:
            root_dir: 包含可见光和红外子目录的根目录
            visible_subdir: 包含可见光图像的子目录名称
            infrared_subdir: 包含红外图像的子目录名称
            image_size: 可选的目标图像尺寸（如果提供，将进行缩放）
            transform: 可选的，应用于可见光图像的变换
            target_transform: 可选的，应用于红外图像的变换
            validate_pairs: 是否在初始化期间验证所有图像对
        """
        # 调用父类的构造函数进行基本初始化
        super().__init__(root_dir, transform, target_transform)
        
        # 构建可见光图像目录的完整路径
        self.visible_dir = self.root_dir / visible_subdir
        # 构建红外图像目录的完整路径
        self.infrared_dir = self.root_dir / infrared_subdir
        # 存储目标图像尺寸
        self.image_size = image_size
        
        # 验证目录结构
        # 检查可见光图像目录是否存在
        if not self.visible_dir.exists():
            raise FileNotFoundError(f"Visible directory not found: {self.visible_dir}")
        # 检查红外图像目录是否存在
        if not self.infrared_dir.exists():
            raise FileNotFoundError(f"Infrared directory not found: {self.infrared_dir}")
        
        # 查找所有有效的图像对
        self.valid_pairs = self._find_valid_pairs(validate_pairs)
        
        # 如果没有找到有效的图像对，则抛出错误
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid visible-infrared pairs found in the dataset")
        
        # 打印找到的有效图像对数量
        print(f"Found {len(self.valid_pairs)} valid visible-infrared pairs")
    
    def _find_valid_pairs(self, validate: bool = True) -> List[Tuple[Path, Path]]:
        """
        在数据集中查找所有有效的可见光-红外图像对。
        
        Args:
            validate: 是否验证每个图像对
            
        Returns:
            (visible_path, infrared_path) 元组的列表
        """
        # 定义可见光图像的常见文件扩展名
        visible_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}
        visible_files = []
        
        # 遍历所有可能的扩展名，查找可见光目录下的图像文件
        for ext in visible_extensions:
            # 查找小写扩展名的文件
            visible_files.extend(self.visible_dir.glob(f"*{ext}"))
            # 查找大写扩展名的文件
            visible_files.extend(self.visible_dir.glob(f"*{ext.upper()}"))
        
        valid_pairs = []
        
        # 遍历所有找到的可见光文件
        for visible_path in visible_files:
            # 根据可见光图像的文件名构建对应的红外图像路径
            infrared_path = self.infrared_dir / visible_path.name
            
            # 如果红外图像路径不存在，尝试不同的扩展名
            if not infrared_path.exists():
                # 获取可见光图像的文件名（不含扩展名）
                stem = visible_path.stem
                found = False
                # 遍历所有可能的扩展名，尝试找到匹配的红外图像
                for ext in visible_extensions:
                    candidate = self.infrared_dir / f"{stem}{ext}"
                    # 如果找到匹配的红外图像
                    if candidate.exists():
                        infrared_path = candidate
                        found = True
                        break
                
                # 如果尝试所有扩展名后仍未找到对应的红外图像，则跳过此可见光图像
                if not found:
                    continue
            
            # 如果请求验证，则调用父类的validate_pair方法进行验证
            if validate and not self.validate_pair(visible_path, infrared_path):
                # 如果验证失败，则跳过此图像对
                continue
            
            # 将有效的图像对添加到列表中
            valid_pairs.append((visible_path, infrared_path))
        
        # 对找到的有效图像对进行排序并返回
        return sorted(valid_pairs)
    
    def __len__(self) -> int:
        """返回有效图像对的总数。"""
        return len(self.valid_pairs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从数据集中获取一个可见光-红外图像对。
        
        Args:
            index: 要检索的图像对的索引
            
        Returns:
            (visible_rgb, infrared_rgb) 张量的元组，形状为 (C, H, W)
        """
        # 检查索引是否超出范围
        if index >= len(self.valid_pairs):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.valid_pairs)}")
        
        # 获取指定索引的可见光和红外图像路径
        visible_path, infrared_path = self.valid_pairs[index]
        
        # 将图像加载为张量
        visible_tensor = self._load_image_as_tensor(visible_path)
        infrared_tensor = self._load_image_as_tensor(infrared_path)
        
        # 如果指定了图像尺寸，则进行缩放
        if self.image_size is not None:
            import torch.nn.functional as F
            # 对可见光张量进行插值缩放
            visible_tensor = F.interpolate(
                visible_tensor.unsqueeze(0), # 增加一个批次维度
                size=(self.image_size, self.image_size), # 目标尺寸
                mode='bilinear', # 双线性插值模式
                align_corners=False # 不对齐角点
            ).squeeze(0) # 移除批次维度
            # 对红外张量进行插值缩放
            infrared_tensor = F.interpolate(
                infrared_tensor.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # 如果提供了变换，则应用于可见光张量
        if self.transform is not None:
            visible_tensor = self.transform(visible_tensor)
        
        # 如果提供了目标变换，则应用于红外张量
        if self.target_transform is not None:
            infrared_tensor = self.target_transform(infrared_tensor)
        
        return visible_tensor, infrared_tensor
    
    def get_metadata(self, index: int) -> Dict[str, Any]:
        """
        获取特定图像对的元数据。
        
        Args:
            index: 图像对的索引
            
        Returns:
            包含文件路径和其他元数据的字典
        """
        # 检查索引是否超出范围
        if index >= len(self.valid_pairs):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.valid_pairs)}")
        
        # 获取指定索引的可见光和红外图像路径
        visible_path, infrared_path = self.valid_pairs[index]
        
        # 返回包含图像路径、文件名、配对ID和索引的字典
        return {
            'visible_path': str(visible_path),
            'infrared_path': str(infrared_path),
            'visible_filename': visible_path.name,
            'infrared_filename': infrared_path.name,
            'pair_id': visible_path.stem, # 使用可见光图像的文件名（不含扩展名）作为配对ID
            'index': index
        }
    
    def get_pair_paths(self, index: int) -> Tuple[Path, Path]:
        """
        获取特定图像对的文件路径。
        
        Args:
            index: 图像对的索引
            
        Returns:
            (visible_path, infrared_path) 的元组
        """
        # 检查索引是否超出范围
        if index >= len(self.valid_pairs):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.valid_pairs)}")
        
        # 返回指定索引的可见光和红外图像路径
        return self.valid_pairs[index]