# Requirements Document

## Introduction

This feature involves migrating the current wandb (Weights & Biases) logging system to use TensorBoard for experiment tracking and visualization. The project currently uses wandb for logging training metrics, test metrics, images, and hyperparameters across multiple trainer classes (IPF_DBDSB, IPF_DSB, IPF_RF). The migration should maintain all existing logging functionality while replacing the wandb backend with TensorBoard.

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to use TensorBoard instead of wandb for logging, so that I can have a local, self-hosted logging solution without external dependencies.

#### Acceptance Criteria

1. WHEN the system starts training THEN it SHALL create TensorBoard log files instead of wandb runs
2. WHEN metrics are logged THEN the system SHALL write them to TensorBoard format
3. WHEN images are logged THEN the system SHALL save them in TensorBoard-compatible format
4. WHEN hyperparameters are logged THEN the system SHALL record them in TensorBoard

### Requirement 2

**User Story:** As a developer, I want to maintain the same logging interface, so that existing trainer code doesn't need to be modified.

#### Acceptance Criteria

1. WHEN the logger is instantiated THEN it SHALL provide the same methods as the current WandbLogger
2. WHEN log_metrics is called THEN it SHALL accept the same parameters as the wandb version
3. WHEN log_image is called THEN it SHALL accept the same parameters as the wandb version
4. WHEN log_hyperparams is called THEN it SHALL accept the same parameters as the wandb version

### Requirement 3

**User Story:** As a user, I want to configure TensorBoard logging through the same configuration system, so that I can easily switch between logging backends.

#### Acceptance Criteria

1. WHEN the config specifies "TensorBoard" as LOGGER THEN the system SHALL use TensorBoard logging
2. WHEN TensorBoard is selected THEN the system SHALL create appropriate log directories
3. WHEN the system runs THEN it SHALL support the same configuration options for log directories and naming
4. WHEN multiple experiments run THEN each SHALL have separate TensorBoard log directories

### Requirement 4

**User Story:** As a researcher, I want to remove wandb dependencies, so that the project has fewer external requirements and can run in environments without wandb access.

#### Acceptance Criteria

1. WHEN the system is installed THEN it SHALL NOT require wandb packages
2. WHEN TensorBoard logging is used THEN the system SHALL NOT import wandb modules
3. WHEN the environment lacks wandb THEN the system SHALL still function normally with TensorBoard
4. WHEN dependencies are checked THEN wandb SHALL be listed as optional, not required

### Requirement 5

**User Story:** As a developer, I want to maintain backward compatibility, so that existing configurations can still work with minimal changes.

#### Acceptance Criteria

1. WHEN old configs specify "Wandb" THEN the system SHALL provide clear migration guidance
2. WHEN CSV logging is used THEN it SHALL continue to work unchanged
3. WHEN no logging is specified THEN the system SHALL default to an appropriate option
4. WHEN invalid logger types are specified THEN the system SHALL provide helpful error messages