from pytorch_lightning.loggers.csv_logs import CSVLogger as _CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger as _TensorBoardLogger
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import io
import os
import time
import threading
from collections import deque
from typing import Dict, List, Any, Optional, Union

class Logger:
    def log_metrics(self, metric_dict, step=None):
        pass

    def log_hyperparams(self, params):
        pass

    def log_image(self, key, images, **kwargs):
        pass


class CSVLogger(_CSVLogger):
    def log_image(self, key, images, **kwargs):
        pass





class TensorBoardLogger(_TensorBoardLogger):
    LOGGER_JOIN_CHAR = '/'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Performance optimization: batch logging
        self._metrics_batch = deque()
        self._images_batch = deque()
        self._batch_size = 10  # Batch size for metrics
        self._image_batch_size = 5  # Smaller batch for images (memory intensive)
        self._last_flush_time = time.time()
        self._flush_interval = 30.0  # Flush every 30 seconds
        
        # Log rotation settings
        self._max_log_files = 50  # Keep max 50 log files per experiment
        self._log_rotation_enabled = True
        
        # Thread-safe logging
        self._logging_lock = threading.Lock()
        
        # Efficient directory management
        self._ensure_log_directory()
        self._setup_log_rotation()

    def _ensure_log_directory(self):
        """Efficiently create and manage log directories"""
        if hasattr(self, 'log_dir') and self.log_dir:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
                # Set appropriate permissions for better performance
                os.chmod(self.log_dir, 0o755)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not set optimal permissions for log directory: {e}")

    def _setup_log_rotation(self):
        """Setup log rotation for long-running experiments"""
        if not self._log_rotation_enabled or not hasattr(self, 'log_dir'):
            return
            
        try:
            # Get all tensorboard log files in the directory
            log_files = []
            if os.path.exists(self.log_dir):
                for root, dirs, files in os.walk(self.log_dir):
                    for file in files:
                        if file.startswith('events.out.tfevents'):
                            file_path = os.path.join(root, file)
                            log_files.append((file_path, os.path.getctime(file_path)))
            
            # Sort by creation time and remove oldest files if exceeding limit
            if len(log_files) > self._max_log_files:
                log_files.sort(key=lambda x: x[1])  # Sort by creation time
                files_to_remove = log_files[:-self._max_log_files]
                
                for file_path, _ in files_to_remove:
                    try:
                        os.remove(file_path)
                        print(f"Removed old log file: {file_path}")
                    except OSError as e:
                        print(f"Warning: Could not remove old log file {file_path}: {e}")
                        
        except Exception as e:
            print(f"Warning: Log rotation setup failed: {e}")

    def _flush_batches(self, force=False):
        """Flush batched metrics and images to TensorBoard"""
        current_time = time.time()
        should_flush = (
            force or 
            len(self._metrics_batch) >= self._batch_size or
            len(self._images_batch) >= self._image_batch_size or
            (current_time - self._last_flush_time) >= self._flush_interval
        )
        
        if not should_flush:
            return
            
        with self._logging_lock:
            # Flush metrics batch
            if self._metrics_batch:
                try:
                    # Group metrics by step for efficient logging
                    step_groups = {}
                    while self._metrics_batch:
                        metrics, step = self._metrics_batch.popleft()
                        if step not in step_groups:
                            step_groups[step] = {}
                        step_groups[step].update(metrics)
                    
                    # Log grouped metrics
                    for step, grouped_metrics in step_groups.items():
                        super().log_metrics(grouped_metrics, step=step)
                        
                except Exception as e:
                    print(f"Warning: Failed to flush metrics batch: {e}")
                    self._metrics_batch.clear()
            
            # Flush images batch
            if self._images_batch:
                try:
                    while self._images_batch:
                        key, img_tensor, step, caption = self._images_batch.popleft()
                        if hasattr(self.experiment, 'add_image'):
                            self.experiment.add_image(
                                key, 
                                img_tensor, 
                                global_step=step,
                                dataformats='CHW'
                            )
                            
                            if caption:
                                self.experiment.add_text(
                                    f"{key}_caption", 
                                    caption, 
                                    global_step=step
                                )
                                
                except Exception as e:
                    print(f"Warning: Failed to flush images batch: {e}")
                    self._images_batch.clear()
            
            self._last_flush_time = current_time

    def log_metrics(self, metrics, step=None, fb=None):
        """Log scalar metrics to TensorBoard with fb prefixing support and batching optimization"""
        # Handle fb parameter for forward/backward prefixing
        if fb is not None:
            metrics.pop('fb', None)
        else:
            fb = metrics.pop('fb', None)
        
        # Apply fb prefixing if specified
        if fb is not None:
            metrics = {fb + '/' + k: v for k, v in metrics.items()}
        
        # Convert tensors to scalars for better performance
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    processed_metrics[key] = value.item()
                else:
                    # For multi-element tensors, log mean
                    processed_metrics[key] = value.mean().item()
            elif isinstance(value, (int, float, np.number)):
                processed_metrics[key] = float(value)
            else:
                # Skip non-numeric values
                continue
        
        # Add to batch for efficient logging
        if processed_metrics:
            self._metrics_batch.append((processed_metrics, step))
            self._flush_batches()

    def log_image(self, key, images, **kwargs):
        """Log images to TensorBoard with batching optimization"""
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        
        step = kwargs.pop("step", None)
        fb = kwargs.pop("fb", None)
        
        # Validate that all kwargs have the same length as images
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        
        # Create kwarg list for each image
        kwarg_list = [{k: kwargs[k][i] for k in kwargs.keys()} for i in range(n)]
        
        # Apply fb prefixing to key if specified
        log_key = key
        if fb is not None:
            log_key = fb + '/' + key
        
        # Process and batch images for efficient logging
        for i, (img, img_kwargs) in enumerate(zip(images, kwarg_list)):
            try:
                # Handle multiple images by adding index to key
                if n == 1:
                    final_key = log_key
                else:
                    final_key = f"{log_key}_{i}"
                
                # Convert image to tensor format for TensorBoard (optimized)
                img_tensor = self._convert_image_to_tensor_optimized(img)
                
                # Extract caption if provided
                caption = img_kwargs.get('caption', '')
                
                # Add to batch for efficient logging
                self._images_batch.append((final_key, img_tensor, step, caption))
                
            except Exception as e:
                print(f"Warning: Failed to process image {i} for key '{final_key}': {e}")
                continue
        
        # Flush batches if needed
        self._flush_batches()

    def _convert_image_to_tensor_optimized(self, image):
        """Optimized image conversion for better performance"""
        # Handle PIL Image
        if isinstance(image, Image.Image):
            # Convert PIL image to numpy array efficiently
            img_array = np.asarray(image)  # More efficient than np.array()
            
            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = img_array[..., np.newaxis]  # More efficient than expand_dims
            
            # Convert to CHW format (channels first) efficiently
            if img_array.shape[2] == 4:  # RGBA - convert to RGB
                img_array = img_array[..., :3]
            
            # Use torch.from_numpy for zero-copy conversion when possible
            img_tensor = torch.from_numpy(img_array.copy()).permute(2, 0, 1).float()
            
            # Efficient normalization
            if img_tensor.dtype == torch.uint8 or img_tensor.max() > 1.0:
                img_tensor = img_tensor.div_(255.0)  # In-place division for efficiency
                
            return img_tensor
        
        # Handle numpy array
        elif isinstance(image, np.ndarray):
            # Create a copy to avoid modifying original
            img_array = image.copy() if not image.flags.c_contiguous else image
            
            # Handle different array shapes efficiently
            if len(img_array.shape) == 2:  # Grayscale (H, W)
                img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
            elif len(img_array.shape) == 3:
                if img_array.shape[0] in [1, 3, 4]:  # Already in CHW format
                    img_tensor = torch.from_numpy(img_array).float()
                    if img_array.shape[0] == 4:  # RGBA - convert to RGB
                        img_tensor = img_tensor[:3, ...]
                elif img_array.shape[2] in [1, 3, 4]:  # HWC format
                    if img_array.shape[2] == 4:  # RGBA - convert to RGB
                        img_array = img_array[..., :3]
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
                else:
                    raise ValueError(f"Unsupported numpy array shape: {img_array.shape}")
            else:
                raise ValueError(f"Unsupported numpy array shape: {img_array.shape}")
            
            # Efficient normalization
            if img_tensor.max() > 1.0:
                img_tensor.div_(255.0)  # In-place division
                
            return img_tensor
        
        # Handle torch tensor
        elif isinstance(image, torch.Tensor):
            # Avoid unnecessary cloning when possible
            img_tensor = image.float() if image.dtype != torch.float32 else image
            
            # Handle different tensor shapes
            if len(img_tensor.shape) == 2:  # Grayscale (H, W)
                img_tensor = img_tensor.unsqueeze(0)
            elif len(img_tensor.shape) == 3:
                if img_tensor.shape[0] not in [1, 3, 4]:  # Assume HWC format
                    img_tensor = img_tensor.permute(2, 0, 1)
                if img_tensor.shape[0] == 4:  # RGBA - convert to RGB
                    img_tensor = img_tensor[:3, ...]
            else:
                raise ValueError(f"Unsupported tensor shape: {img_tensor.shape}")
            
            # Efficient normalization
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor.div(255.0)  # Create new tensor to avoid modifying original
                
            return img_tensor
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image, numpy.ndarray, or torch.Tensor")

    def _convert_image_to_tensor(self, image):
        """Legacy method for backward compatibility - delegates to optimized version"""
        return self._convert_image_to_tensor_optimized(image)

    def log_hyperparams(self, params):
        """Optimized hyperparameter logging to TensorBoard"""
        try:
            # Convert nested config to flat dictionary for TensorBoard
            flat_params = self._flatten_dict_optimized(params)
            
            # Filter out non-serializable values and convert to appropriate types efficiently
            serializable_params = self._filter_serializable_params(flat_params)
            
            # Log hyperparameters using parent class method
            if serializable_params:
                super().log_hyperparams(serializable_params)
                
        except Exception as e:
            print(f"Warning: Failed to log hyperparameters: {e}")

    def _filter_serializable_params(self, flat_params):
        """Efficiently filter and convert parameters to serializable format"""
        serializable_params = {}
        
        for key, value in flat_params.items():
            try:
                # Handle basic types efficiently
                if isinstance(value, (int, float, str, bool)):
                    serializable_params[key] = value
                elif isinstance(value, torch.Tensor):
                    # Convert tensors to scalars or skip
                    if value.numel() == 1:
                        serializable_params[key] = value.item()
                    elif value.numel() <= 10:  # Small tensors to string
                        serializable_params[key] = str(value.tolist())
                elif isinstance(value, np.ndarray):
                    # Handle numpy arrays
                    if value.size == 1:
                        serializable_params[key] = float(value.item())
                    elif value.size <= 10:  # Small arrays to string
                        serializable_params[key] = str(value.tolist())
                elif isinstance(value, (list, tuple)) and len(value) <= 20:
                    # Convert small lists/tuples to string representation
                    serializable_params[key] = str(value)
                elif value is None:
                    serializable_params[key] = "None"
                elif hasattr(value, '__dict__') and len(str(value)) < 200:
                    # Convert small objects to string
                    serializable_params[key] = str(value)
                # Skip large or complex objects silently
                    
            except Exception:
                # Skip values that can't be serialized
                continue
                
        return serializable_params

    def finalize(self, status: str = "success"):
        """Finalize logging and cleanup resources"""
        try:
            # Flush any remaining batches
            self._flush_batches(force=True)
            
            # Cleanup batches first
            self._metrics_batch.clear()
            self._images_batch.clear()
            
            # Call parent finalize if it exists, with error handling
            try:
                if hasattr(super(), 'finalize'):
                    super().finalize(status)
            except (FileNotFoundError, OSError) as e:
                # Handle cases where temporary directories are already cleaned up
                pass
            except Exception as e:
                print(f"Warning: Error during parent finalization: {e}")
                
        except Exception as e:
            print(f"Warning: Error during TensorBoard logger finalization: {e}")

    def __del__(self):
        """Cleanup when logger is destroyed"""
        try:
            self.finalize("cleanup")
        except Exception:
            pass  # Ignore errors during cleanup

    def _flatten_dict_optimized(self, d, parent_key='', sep='/', max_depth=5):
        """Optimized flattening of nested dictionary with depth limit"""
        if max_depth <= 0:
            return {parent_key: str(d)} if parent_key else {'truncated': str(d)}
            
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            # Limit key length to prevent extremely long keys
            if len(new_key) > 100:
                new_key = new_key[:97] + "..."
                
            if isinstance(v, dict) and len(v) > 0:
                items.extend(self._flatten_dict_optimized(v, new_key, sep=sep, max_depth=max_depth-1).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _flatten_dict(self, d, parent_key='', sep='/'):
        """Legacy method for backward compatibility"""
        return self._flatten_dict_optimized(d, parent_key, sep)