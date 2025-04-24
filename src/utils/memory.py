"""
Memory management utilities for Word2Vec training.
"""

import torch
import numpy as np
from typing import Union

def get_gpu_memory() -> float:
    """Get available GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    return 0

def calculate_dynamic_batch_size(model: torch.nn.Module, device: torch.device, 
                               safety_factor: float = 0.5) -> int:
    """Calculate optimal batch size based on available memory."""
    total_memory = get_gpu_memory()
    if total_memory == 0:
        return 32  # Default batch size if no GPU
    
    # Estimate memory per sample
    sample_size = (model.embeddings[0].embedding_dim * 5 * 4)  # 4 bytes per float
    # Add overhead for model parameters and gradients
    overhead = (model.embeddings[0].embedding_dim * len(model.embeddings) * 4) / 1024  # KB
    available_memory = (total_memory * safety_factor) - overhead
    
    max_batch_size = int(available_memory / sample_size)
    
    # Ensure batch size is a power of 2 and within reasonable limits
    batch_size = min(2 ** int(np.log2(max_batch_size)), 256)
    return max(16, batch_size)  # Minimum batch size of 16

def clear_memory() -> None:
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect() 