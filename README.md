# Word2Vec Implementation with Memory Optimization

This project implements a memory-efficient Word2Vec model using the Continuous Bag of Words (CBOW) architecture. It includes various optimizations to handle large datasets while managing GPU memory constraints.

## Project Structure

```
search-engine/
├── main.py                    # Main entry point
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── data/                      # Data directory
│   └── text8                  # Dataset
├── src/                       # Source code directory
│   ├── __init__.py           # Makes src a Python package
│   ├── data/                 # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocess.py     # Text preprocessing
│   │   └── dataset.py        # Dataset loading and management
│   ├── models/               # Model-related modules
│   │   ├── __init__.py
│   │   ├── word2vec.py      # Word2Vec model implementation
│   │   └── cbow.py          # CBOW model implementation
│   ├── training/             # Training-related modules
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training logic
│   │   └── optimizer.py     # Optimizer configuration
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── memory.py        # Memory management utilities
│       └── logging.py       # Logging configuration
└── tests/                    # Test directory
    ├── __init__.py
    ├── test_data.py
    ├── test_models.py
    └── test_training.py
```

## Features

- Memory-efficient implementation with chunked processing
- Dynamic batch sizing based on available GPU memory
- Mixed precision training
- Gradient checkpointing
- Sparse embeddings
- Weights & Biases integration for experiment tracking

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd search-engine
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your text8 dataset in the `data/` directory.

2. Run the training:
```bash
python main.py
```

## Configuration

The model can be configured by modifying the parameters in `src/config.py`:

- `window_size`: Context window size
- `min_word_freq`: Minimum word frequency threshold
- `max_vocab_size`: Maximum vocabulary size
- `embedding_dim`: Embedding dimension
- `learning_rate`: Learning rate
- `batch_size`: Initial batch size
- `epochs`: Number of training epochs
- `use_sparse`: Whether to use sparse embeddings
- `use_mixed_precision`: Whether to use mixed precision training
- `gradient_accumulation_steps`: Number of gradient accumulation steps
- `chunk_size`: Size of data chunks for processing
- `memory_safety_factor`: Safety factor for memory usage

## Memory Optimizations

- Vocabulary chunking
- Dynamic batch sizing
- Gradient checkpointing
- Mixed precision training
- Sparse embeddings
- Aggressive memory cleanup

## License

This project is licensed under the MIT License - see the LICENSE file for details. 