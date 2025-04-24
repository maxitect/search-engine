import torch
import argparse
import numpy as np
import torch.nn as nn
import re
import os

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        embeds = self.embeddings(context).mean(dim=1)
        return self.linear(embeds)

def get_similar_words(model, word, word2idx, idx2word, top_k=5):
    """Get similar words using cosine similarity"""
    # Clean the input word to match our preprocessing
    word = word.lower()
    
    # Check if word exists in vocabulary
    if word not in word2idx:
        # Try to find similar words by removing special characters
        clean_word = re.sub(r'[^a-z0-9]', '', word)
        if clean_word in word2idx:
            word = clean_word
        else:
            return f"Word '{word}' not in vocabulary. Try another word."
    
    word_idx = word2idx[word]
    word_embedding = model.embeddings.weight[word_idx]
    
    # Calculate cosine similarity with all words
    similarities = torch.nn.functional.cosine_similarity(
        word_embedding.unsqueeze(0),
        model.embeddings.weight,
        dim=1
    )
    
    # Get top k similar words (excluding the word itself)
    top_k_indices = torch.topk(similarities, k=top_k+1)[1][1:]  # +1 to exclude the word itself
    similar_words = [idx2word[idx.item()] for idx in top_k_indices]
    
    # Filter out special tokens from results
    similar_words = [w for w in similar_words if not w.startswith('<')]
    
    return similar_words

def load_model(model_path="data/word2vec/word2vec_cbow_final.pt"):
    """Load the trained model and vocabulary"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    # Load the saved embeddings and vocabulary
    try:
        # First try with weights_only=True and allow numpy arrays
        torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    except Exception as e:
        # If that fails, try with weights_only=False (old behavior)
        print("Warning: Loading with weights_only=True failed. Trying with weights_only=False...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    if 'embeddings' in checkpoint:
        # Final model format
        embeddings = checkpoint['embeddings']
        vocab = checkpoint['vocab']
        
        # Create word to index mapping
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        # Create model and load embeddings
        model = CBOW(len(vocab), embedding_dim=300)  # Match the embedding dimension used in training
        model.embeddings.weight.data = torch.FloatTensor(embeddings)
        model.eval()
        
    else:
        # Checkpoint format
        vocab_path = model_path.replace(model_path.split('/')[-1], 'word2vec_cbow_final.pt')
        try:
            vocab_data = torch.load(vocab_path, map_location=torch.device('cpu'), weights_only=True)
        except Exception as e:
            print("Warning: Loading vocabulary with weights_only=True failed. Trying with weights_only=False...")
            vocab_data = torch.load(vocab_path, map_location=torch.device('cpu'), weights_only=False)
            
        vocab = vocab_data['vocab']
        
        # Create word to index mapping
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        # Create model and load state
        model = CBOW(len(vocab), embedding_dim=300)  # Match the embedding dimension used in training
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    return model, word2idx, idx2word

def main():
    parser = argparse.ArgumentParser(description='Test Word2Vec model for word similarities')
    parser.add_argument('--word', type=str, required=True, help='Input word to find similar words')
    parser.add_argument('--model', type=str, default="data/word2vec/word2vec_cbow_final.pt", 
                        help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Number of similar words to return')
    
    args = parser.parse_args()
    
    try:
        # Load model and vocabulary
        model, word2idx, idx2word = load_model(args.model)
        
        # Find similar words
        similar = get_similar_words(model, args.word, word2idx, idx2word, args.top_k)
        
        # Print results
        if isinstance(similar, str):
            print(similar)  # Error message
        else:
            print(f"Words similar to '{args.word}':")
            for i, word in enumerate(similar, 1):
                print(f"{i}. {word}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure you have trained the model first using train_word2vec.py")

if __name__ == "__main__":
    main()