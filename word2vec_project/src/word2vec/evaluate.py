import torch
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm

def evaluate_word_similarity(
    model: torch.nn.Module,
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    test_words: List[str],
    top_k: int = 5
) -> Dict[str, List[Tuple[str, float]]]:
    """Evaluate word similarity using the trained model."""
    results = {}
    
    for word in tqdm(test_words, desc="Evaluating word similarities"):
        if word not in word_to_idx:
            continue
            
        word_idx = word_to_idx[word]
        similar_words = model.get_similar_words(word_idx, top_k=top_k)
        
        # Convert indices to words and format results
        word_results = []
        for idx, score in similar_words:
            if idx in idx_to_word:
                word_results.append((idx_to_word[idx], score))
        
        results[word] = word_results
    
    return results

def evaluate_analogy_task(
    model: torch.nn.Module,
    word_to_idx: Dict[str, int],
    idx_to_word: Dict[int, str],
    analogies: List[Tuple[str, str, str, str]],
    top_k: int = 5
) -> float:
    """Evaluate the model on word analogy tasks."""
    correct = 0
    total = 0
    
    for a, b, c, d in tqdm(analogies, desc="Evaluating analogies"):
        if not all(word in word_to_idx for word in [a, b, c, d]):
            continue
            
        # Get word vectors
        a_vec = model.get_word_vector(word_to_idx[a])
        b_vec = model.get_word_vector(word_to_idx[b])
        c_vec = model.get_word_vector(word_to_idx[c])
        
        # Compute analogy vector
        analogy_vec = b_vec - a_vec + c_vec
        
        # Find most similar words
        similarities = torch.matmul(model.embeddings.weight, analogy_vec.squeeze())
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k)
        
        # Check if correct word is in top k
        if word_to_idx[d] in top_k_indices:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy

def main():
    # This will be implemented when we have the actual model and data
    pass

if __name__ == "__main__":
    main() 