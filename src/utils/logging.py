"""
Logging configuration for Word2Vec training.
"""

import logging
import os
from datetime import datetime

def setup_logging() -> None:
    """Setup logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set logging level for specific modules
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.INFO) 