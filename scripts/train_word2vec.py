import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.word2vec.model import Word2Vec
from src.word2vec.train import train_word2vec
from src.data.preprocess import preprocess_file
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Word2Vec model')
    parser.add_argument('--embedding-dim', type=int, default=100,
                       help='Dimension of word embeddings')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    args = parser.parse_args()
    
    # Preprocess the text data
    input_path = "data/processed/combined.txt"
    output_path = "data/processed/preprocessed.txt"
    preprocess_file(input_path, output_path)
    
    # TODO: Load preprocessed data and create training pairs
    # This will be implemented when we have the actual data
    
    # Initialize model
    vocab_size = 10000  # This will be determined from the data
    model = Word2Vec(vocab_size=vocab_size, embedding_dim=args.embedding_dim)
    
    # Train model
    train_word2vec(
        model=model,
        train_data=[],  # This will be the training pairs
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main() 