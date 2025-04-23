import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.word2vec.model import Word2Vec
from src.word2vec.train import train_word2vec
from src.data.preprocess import preprocess_file
from src.data.combine_datasets import combine_datasets
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
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/word2vec/trained", exist_ok=True)
    os.makedirs("models/word2vec/checkpoints", exist_ok=True)
    
    # Check if input files exist
    input_path = "data/processed/combined.txt"
    if not os.path.exists(input_path):
        print("Input file not found. Creating combined dataset...")
        # TODO: Load actual datasets here
        # For now, create a small sample dataset
        sample_text = """
        Machine learning is a field of artificial intelligence that focuses on developing algorithms 
        that can learn from and make predictions on data. Neural networks are a type of machine learning 
        model inspired by the structure of the human brain. Deep learning is a subset of machine learning 
        that uses multiple layers of neural networks to learn complex patterns in data.
        """
        with open(input_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print(f"Created sample dataset at {input_path}")
    
    # Preprocess the text data
    output_path = "data/processed/preprocessed.txt"
    print("Preprocessing text data...")
    preprocess_file(input_path, output_path)
    
    # TODO: Load preprocessed data and create training pairs
    # This will be implemented when we have the actual data
    print("Note: Using sample data for demonstration. Replace with actual data for real training.")
    
    # Initialize model
    vocab_size = 10000  # This will be determined from the data
    model = Word2Vec(vocab_size=vocab_size, embedding_dim=args.embedding_dim)
    
    # Train model
    print("\nStarting training...")
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