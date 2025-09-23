# Design Document

## Overview

This design outlines the migration from wandb to TensorBoard logging in the DSBM (Diffusion Schrödinger Bridge Models) project. The migration will replace the current WandbLogger with a TensorBoardLogger while maintaining the same interface and functionality. The design ensures minimal code changes to existing trainer classes while providing equivalent logging capabilities.

## Architecture

### Current Architecture
- `WandbLogger` class inherits from PyTorch Lightning's `WandbLogger`
- Logger instantiation handled in `get_logger()` function in `config_getters.py`
- Configuration driven by `LOGGER` parameter in config files
- Supports metrics, images, and hyperparameter logging

### New Architecture
- `TensorBoardLogger` class will inherit from PyTorch Lightning's `TensorBoardLogger`
- Same interface as `WandbLogger` to maintain compatibility
- Logger selection enhanced to support "TensorBoard" option
- TensorBoard-specific configuration options added

## Components and Interfaces

### 1. TensorBoardLogger Class

**Location:** `bridge/runners/logger.py`

**Interface:**
```python
class TensorBoardLogger(_TensorBoardLogger):
    LOGGER_JOIN_CHAR = '/'
    
    def log_metrics(self, metrics, step=None, fb=None):
        """Log scalar metrics to TensorBoard"""
        
    def log_image(self, key, images, **kwargs):
        """Log images to TensorBoard"""
        
    def log_hyperparams(self, params):
        """Log hyperparameters to TensorBoard"""
```

**Key Features:**
- Maintains same method signatures as WandbLogger
- Handles fb (forward/backward) prefixing like wandb version
- Converts image formats for TensorBoard compatibility
- Manages step tracking consistently

### 2. Configuration Updates

**Location:** `bridge/runners/config_getters.py`

**Changes:**
- Add `TENSORBOARD_TAG = 'TensorBoard'` constant
- Extend `get_logger()` function to handle TensorBoard option
- Add TensorBoard-specific configuration parsing
- Maintain backward compatibility with existing configs

**New Configuration Options:**
```yaml
LOGGER: TensorBoard
tensorboard_log_dir: ./tensorboard_logs
tensorboard_name: experiment_name
```

### 3. Dependency Management

**Location:** `bridge.def`, `requirements.txt` (if exists)

**Changes:**
- Remove wandb from required dependencies
- Add tensorboard as required dependency
- Update import statements to be conditional

## Data Models

### Logging Data Flow

1. **Metrics Logging:**
   - Input: `dict` of metric names and values
   - Processing: Apply fb prefixing, step management
   - Output: TensorBoard scalar logs

2. **Image Logging:**
   - Input: List of images, captions, step info
   - Processing: Convert to TensorBoard image format
   - Output: TensorBoard image logs

3. **Hyperparameter Logging:**
   - Input: Configuration dictionary
   - Processing: Flatten nested configs, type conversion
   - Output: TensorBoard hyperparameter logs

### Configuration Schema

```python
tensorboard_config = {
    'log_dir': str,           # Base directory for logs
    'name': str,              # Experiment name
    'version': Optional[str], # Version/run identifier
    'prefix': Optional[str],  # Prefix for run name
}
```

## Error Handling

### 1. Missing Dependencies
- **Error:** TensorBoard not installed
- **Handling:** Provide clear installation instructions
- **Fallback:** Suggest CSV logger as alternative

### 2. Directory Permissions
- **Error:** Cannot create log directory
- **Handling:** Create parent directories, check permissions
- **Fallback:** Use temporary directory with warning

### 3. Image Format Issues
- **Error:** Unsupported image format
- **Handling:** Convert to supported format (PNG/JPEG)
- **Fallback:** Log error message, skip image

### 4. Configuration Errors
- **Error:** Invalid logger type specified
- **Handling:** Provide helpful error message with valid options
- **Fallback:** Default to CSV logger with warning

## Testing Strategy

### 1. Unit Tests
- **TensorBoardLogger Methods:** Test each logging method independently
- **Configuration Parsing:** Test logger instantiation with various configs
- **Error Handling:** Test error conditions and fallbacks

### 2. Integration Tests
- **Trainer Integration:** Test with actual trainer classes
- **Multi-process:** Test with distributed training setup
- **File System:** Test log file creation and structure

### 3. Compatibility Tests
- **Interface Compatibility:** Ensure same method signatures work
- **Data Format:** Verify logged data matches expected format
- **Performance:** Compare logging overhead with wandb

### 4. Migration Tests
- **Config Migration:** Test old configs with new system
- **Data Continuity:** Ensure metrics are logged correctly
- **Visualization:** Verify TensorBoard displays data correctly

## Implementation Phases

### Phase 1: Core TensorBoard Logger
- Implement `TensorBoardLogger` class
- Add basic metrics and hyperparameter logging
- Update configuration system

### Phase 2: Image Logging
- Implement image logging functionality
- Handle image format conversions
- Test with trainer image logging calls

### Phase 3: Configuration and Integration
- Update config files and documentation
- Add error handling and fallbacks
- Integration testing with trainers

### Phase 4: Cleanup and Optimization
- Remove wandb dependencies
- Optimize performance
- Add comprehensive documentation

## Migration Path

### For Users:
1. Update config files: Change `LOGGER: Wandb` to `LOGGER: TensorBoard`
2. Install tensorboard: `pip install tensorboard`
3. Update log directory configs if needed
4. Launch TensorBoard: `tensorboard --logdir=./tensorboard_logs`

### For Developers:
1. Import changes are handled automatically
2. Same logging interface maintained
3. No trainer code changes required
4. Configuration updates may be needed

## Performance Considerations

### TensorBoard vs Wandb:
- **Local Storage:** TensorBoard stores logs locally, reducing network overhead
- **Memory Usage:** Similar memory footprint for logging operations
- **Disk Usage:** TensorBoard files may be larger but more accessible
- **Startup Time:** Faster startup without wandb authentication

### Optimization Strategies:
- Batch image logging to reduce I/O
- Use efficient image formats (PNG for screenshots, JPEG for photos)
- Implement log rotation for long-running experiments
- Cache frequently accessed configuration values