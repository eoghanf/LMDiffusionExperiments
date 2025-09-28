"""
Utility functions for NoProp training.
"""

import torch
import numpy as np
import random
import time
from typing import Optional


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_device_info():
    """Print information about available devices."""
    device = get_device()
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format time in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


class Timer:
    """Simple timer utility for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        if self.end_time is None:
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time
    
    def elapsed_str(self) -> str:
        """Get elapsed time as a formatted string."""
        return format_time(self.elapsed())


def save_checkpoint(model: torch.nn.Module, optimizers: dict, epoch: int, 
                   best_accuracy: float, filepath: str, **kwargs):
    """
    Save training checkpoint.
    
    Args:
        model: The model to save
        optimizers: Dictionary of optimizers
        epoch: Current epoch
        best_accuracy: Best accuracy achieved so far
        filepath: Path to save the checkpoint
        **kwargs: Additional data to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizers': {k: opt.state_dict() for k, opt in optimizers.items()},
        'best_accuracy': best_accuracy,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizers: Optional[dict] = None):
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: Model to load state into
        optimizers: Optional dictionary of optimizers to load state into
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer states if provided
    if optimizers is not None and 'optimizers' in checkpoint:
        for k, opt in optimizers.items():
            if k in checkpoint['optimizers']:
                opt.load_state_dict(checkpoint['optimizers'][k])
    
    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update statistics with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_training_header(config):
    """Print formatted training configuration header."""
    print(f"=== NoProp {config.dataset.upper()} Training ===")
    
    if hasattr(config, '_config_path') and config._config_path:
        print(f"Config file: {config._config_path}")
    
    print(f"Dataset: {config.dataset}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs per layer: {config.epochs_per_layer}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")
    print(f"Timesteps: {config.timesteps}")
    if hasattr(config, 'eta'):
        print(f"Eta (η): {config.eta}")
    print()


def print_model_info(model: torch.nn.Module):
    """Print model parameter information."""
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print()


def reshape_sequence_data(input_ids: torch.Tensor, targets: torch.Tensor, max_seq_len: int):
    """
    Reshape 1D sequence data to proper format for language modeling.
    
    Args:
        input_ids: 1D tensor of token ids from data generator
        targets: 1D tensor of target token ids from data generator  
        max_seq_len: Maximum sequence length to use
        
    Returns:
        tuple of (input_ids, targets) reshaped to (1, seq_len)
    """
    # The distributed_data_generator returns 1D sequences of length batch_size
    # We need to reshape to proper sequences for language modeling
    seq_len = min(max_seq_len, input_ids.size(0))
    if input_ids.size(0) < max_seq_len:
        # Pad if necessary
        pad_size = max_seq_len - input_ids.size(0)
        input_ids = torch.cat([input_ids, torch.zeros(pad_size, dtype=input_ids.dtype, device=input_ids.device)])
        targets = torch.cat([targets, torch.ones(pad_size, dtype=targets.dtype, device=targets.device) * -1])  # Use -1 for ignore_index
    
    # Reshape to (1, seq_len) since we have one long sequence
    input_ids = input_ids[:max_seq_len].unsqueeze(0)  # (1, seq_len)
    targets = targets[:max_seq_len].unsqueeze(0)  # (1, seq_len)
    
    return input_ids, targets


def create_triangular_subsequences_batched(seq, padding_value=-1, batch_size=None):
    seq_length = len(seq)

    if batch_size is None:
        batch_size = seq_length

    result = torch.full((seq_length, seq_length), padding_value, dtype=seq.dtype)

    for start_idx in range(0, seq_length, batch_size):
        end_idx = min(start_idx + batch_size, seq_length)

        # Process batch of rows
        batch_full_matrix = seq.unsqueeze(0).repeat(end_idx - start_idx, 1)

        # Create triangular mask for this batch
        batch_size_actual = end_idx - start_idx
        rows = torch.arange(start_idx, end_idx).unsqueeze(1)  # Actual row indices
        cols = torch.arange(seq_length).unsqueeze(0)
        mask = cols <= rows

        # Apply mask
        result[start_idx:end_idx] = torch.where(
            mask,
            batch_full_matrix,
            padding_value
        )
