from pytorch_lightning.loggers import CSVLogger as _CSVLogger, WandbLogger as _WandbLogger, TensorBoardLogger as _TensorBoardLogger
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import io

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


class WandbLogger(_WandbLogger):
    LOGGER_JOIN_CHAR = '/'

    def log_metrics(self, metrics, step=None, fb=None):
        if fb is not None:
            metrics.pop('fb', None)
        else:
            fb = metrics.pop('fb', None)
        if fb is not None:
            metrics = {fb + '/' + k: v for k, v in metrics.items()}
        super().log_metrics(metrics, step=step)

    def log_image(self, key, images, **kwargs):
        if not isinstance(images, list):
            raise TypeError(f'Expected a list as "images", found {type(images)}')
        step = kwargs.pop("step", None)
        fb = kwargs.pop("fb", None)
        n = len(images)
        for k, v in kwargs.items():
            if len(v) != n:
                raise ValueError(f"Expected {n} items but only found {len(v)} for {k}")
        kwarg_list = [{k: kwargs[k][i] for k in kwargs.keys()} for i in range(n)]
        if n == 1:
            metrics = {key: wandb.Image(images[0], **kwarg_list[0])}
        else:
            metrics = {key: [wandb.Image(img, **kwarg) for img, kwarg in zip(images, kwarg_list)]}
        self.log_metrics(metrics, step=step, fb=fb)


class TensorBoardLogger(_TensorBoardLogger):
    LOGGER_JOIN_CHAR = '/'

    def log_metrics(self, metrics, step=None, fb=None):
        """Log scalar metrics to TensorBoard with fb prefixing support identical to WandbLogger"""
        # Handle fb parameter the same way as WandbLogger
        if fb is not None:
            metrics.pop('fb', None)
        else:
            fb = metrics.pop('fb', None)
        
        # Apply fb prefixing if specified
        if fb is not None:
            metrics = {fb + '/' + k: v for k, v in metrics.items()}
        
        # Log metrics using parent class method
        super().log_metrics(metrics, step=step)

    def log_image(self, key, images, **kwargs):
        """Log images to TensorBoard with same interface as WandbLogger"""
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
        
        # Convert and log images
        for i, (img, img_kwargs) in enumerate(zip(images, kwarg_list)):
            # Handle multiple images by adding index to key
            if n == 1:
                final_key = log_key
            else:
                final_key = f"{log_key}_{i}"
            
            # Convert image to tensor format for TensorBoard
            img_tensor = self._convert_image_to_tensor(img)
            
            # Extract caption if provided
            caption = img_kwargs.get('caption', '')
            
            # Log image using the experiment's SummaryWriter
            if hasattr(self.experiment, 'add_image'):
                self.experiment.add_image(
                    final_key, 
                    img_tensor, 
                    global_step=step,
                    dataformats='CHW'
                )
                
                # Log caption as text if provided
                if caption:
                    self.experiment.add_text(
                        f"{final_key}_caption", 
                        caption, 
                        global_step=step
                    )

    def _convert_image_to_tensor(self, image):
        """Convert various image formats to tensor format for TensorBoard"""
        # Handle PIL Image
        if isinstance(image, Image.Image):
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=2)
            
            # Convert to CHW format (channels first)
            if img_array.shape[2] == 1:  # Grayscale
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            elif img_array.shape[2] == 3:  # RGB
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            elif img_array.shape[2] == 4:  # RGBA - convert to RGB
                img_array = img_array[:, :, :3]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            else:
                raise ValueError(f"Unsupported image format with {img_array.shape[2]} channels")
            
            # Normalize to [0, 1] if values are in [0, 255]
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
                
            return img_tensor
        
        # Handle numpy array
        elif isinstance(image, np.ndarray):
            # Handle different array shapes
            if len(image.shape) == 2:  # Grayscale (H, W)
                img_tensor = torch.from_numpy(image).unsqueeze(0).float()  # Add channel dimension
            elif len(image.shape) == 3:
                if image.shape[0] in [1, 3, 4]:  # Already in CHW format
                    img_tensor = torch.from_numpy(image).float()
                    if image.shape[0] == 4:  # RGBA - convert to RGB
                        img_tensor = img_tensor[:3, :, :]
                elif image.shape[2] in [1, 3, 4]:  # HWC format
                    if image.shape[2] == 4:  # RGBA - convert to RGB
                        image = image[:, :, :3]
                    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
                else:
                    raise ValueError(f"Unsupported numpy array shape: {image.shape}")
            else:
                raise ValueError(f"Unsupported numpy array shape: {image.shape}")
            
            # Normalize to [0, 1] if values are in [0, 255]
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
                
            return img_tensor
        
        # Handle torch tensor
        elif isinstance(image, torch.Tensor):
            img_tensor = image.clone().float()
            
            # Handle different tensor shapes
            if len(img_tensor.shape) == 2:  # Grayscale (H, W)
                img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
            elif len(img_tensor.shape) == 3:
                if img_tensor.shape[0] not in [1, 3, 4]:  # Assume HWC format
                    img_tensor = img_tensor.permute(2, 0, 1)
                if img_tensor.shape[0] == 4:  # RGBA - convert to RGB
                    img_tensor = img_tensor[:3, :, :]
            else:
                raise ValueError(f"Unsupported tensor shape: {img_tensor.shape}")
            
            # Normalize to [0, 1] if values are in [0, 255]
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
                
            return img_tensor
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image, numpy.ndarray, or torch.Tensor")

    def log_hyperparams(self, params):
        """Log hyperparameters to TensorBoard"""
        # Convert nested config to flat dictionary for TensorBoard
        flat_params = self._flatten_dict(params)
        
        # Filter out non-serializable values and convert to appropriate types
        serializable_params = {}
        for key, value in flat_params.items():
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
        
        # Log hyperparameters using parent class method
        super().log_hyperparams(serializable_params)

    def _flatten_dict(self, d, parent_key='', sep='/'):
        """Flatten nested dictionary for hyperparameter logging"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)