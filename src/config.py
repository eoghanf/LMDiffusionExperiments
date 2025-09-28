"""
Configuration management for NoProp training experiments.
Loads configurations from YAML files in experiment_configs/
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union


@dataclass
class NoPropConfig:
    """Configuration class for NoProp training."""
    
    # Dataset settings
    dataset: str = "mnist"
    batch_size: int = 128
    data_path: str = "./data"
    num_workers: int = 4
    augment: bool = False
    
    # Model architecture
    num_layers: int = 10
    
    # Training parameters
    outer_loops: int = 100
    batches_per_layer: int = 50
    layer_lr: float = 0.001
    embedding_lr: float = 0.01
    embedding_weight_decay: float = 0.001
    layer_weight_decay: float = 0.001
    layer_optimizer: str = "AdamW"
    embedding_optimizer: str = "Adagrad"
    
    # Language model specific parameters
    embed_dim: int = 512
    num_heads: int = 8
    ff_dim: int = 2048
    max_seq_len: int = 128
    dropout: float = 0.1
    
    # NoProp specific parameters
    timesteps: int = 10
    eta: float = 0.1
    
    # Noise schedule parameters
    noise_schedule_type: str = "cosine"  # "cosine" or "linear"
    noise_schedule_min: float = 0.001  # Minimum noise level
    noise_schedule_max: float = 0.999  # Maximum noise level
    
    # Optimization settings
    grad_clip_max_norm: float = 1.0
    
    # Logging and saving
    log_interval: int = 100
    save_best: bool = True
    save_final: bool = True
    save_checkpoints: bool = False
    detailed_logging: bool = True
    validation_batches_per_log: int = 5
    best_model_path: str = "checkpoints/best_model.pt"
    final_model_path: str = "checkpoints/final_model.pt"
    checkpoint_dir: str = "checkpoints"
    
    # Early stopping (disabled for language modeling)
    early_stopping: bool = False
    early_stopping_accuracy: float = 99.5
    
    # Reproducibility
    seed: int = 42
    
    # Store original config dict
    _config_dict: Dict[str, Any] = field(default_factory=dict, repr=False)
    _config_path: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.timesteps != self.num_layers:
            print(f"Warning: timesteps ({self.timesteps}) != num_layers ({self.num_layers})")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'NoPropConfig':
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration object loaded from YAML
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        # Create instance with config values
        instance = cls(**config_dict)
        instance._config_dict = config_dict.copy()
        instance._config_path = str(config_path)
        
        return instance
    
    
    
    def update(self, **kwargs) -> 'NoPropConfig':
        """
        Create a new config with updated values.
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            New configuration object with updated values
        """
        # Get current values as dict
        current_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                current_dict[key] = value
        
        # Update with new values
        current_dict.update(kwargs)
        
        # Create new instance
        new_config = self.__class__(**current_dict)
        new_config._config_dict = self._config_dict.copy()
        new_config._config_path = self._config_path
        
        return new_config
    
    def print_config(self):
        """Print configuration in a readable format."""
        print("=== Configuration ===")
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                print(f"{key}: {value}")
        print()


def load_config(config_path: Union[str, Path], **overrides) -> NoPropConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        **overrides: Override any configuration values
        
    Returns:
        Configuration object
    """
    config = NoPropConfig.from_yaml(config_path)
    
    # Apply any overrides
    if overrides:
        config = config.update(**overrides)
    
    return config