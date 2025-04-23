import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.word2vec.model import Word2Vec
from src.word2vec.evaluate import evaluate_word_similarity, evaluate_analogy_task
import torch
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description='Test Word2Vec embeddings')
    parser.add_argument('--model-path', type=str, default='models/word2vec/trained/final_model.pth',
                       help='Path to trained model')
    parser.add_argument('--vocab-path', type=str, default='data/processed/vocab/word_to_idx.json',
                       help='Path to vocabulary mapping')
    args = parser.parse_args()
    
    # Load vocabulary
    with open(args.vocab_path, 'r') as f:
        word_to_idx = json.load(f)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Initialize model
    vocab_size = len(word_to_idx)
    model = Word2Vec(vocab_size=vocab_size, embedding_dim=100)  # embedding_dim should match trained model
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Test words for similarity evaluation
    test_words = [
        "machine", "learning", "neural", "network",
        "artificial", "intelligence", "data", "science"
    ]
    
    # Evaluate word similarities
    print("\nEvaluating word similarities:")
    similarity_results = evaluate_word_similarity(
        model=model,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        test_words=test_words
    )
    
    for word, similar_words in similarity_results.items():
        print(f"\nWords similar to '{word}':")
        for similar_word, score in similar_words:
            print(f"  {similar_word}: {score:.4f}")
    
    # Test analogies
    analogies = [
        ("king", "queen", "man", "woman"),
        ("paris", "france", "rome", "italy"),
        ("cat", "cats", "dog", "dogs")
    ]
    
    # Evaluate analogies
    print("\nEvaluating word analogies:")
    analogy_accuracy = evaluate_analogy_task(
        model=model,
        word_to_idx=word_to_idx,
        idx_to_word=idx_to_word,
        analogies=analogies
    )
    
    print(f"Analogy task accuracy: {analogy_accuracy:.4f}")

if __name__ == "__main__":
    main() 