import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import os
from tqdm import tqdm
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
import random

def load_text8_data(text8_path):
    """Load text8 dataset."""
    with open(text8_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text.split()

def get_bert_embeddings(text, tokenizer, model, device, batch_size=32):
    """Get BERT embeddings for a list of words."""
    embeddings = {}
    model.eval()
    
    # Process words in batches
    for i in tqdm(range(0, len(text), batch_size), desc="Getting BERT embeddings"):
        batch_words = text[i:i + batch_size]
        
        # Tokenize the words
        encoded = tokenizer(batch_words, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # Use the [CLS] token embedding (first token) as the word embedding
            word_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # Store embeddings
        for word, embedding in zip(batch_words, word_embeddings):
            embeddings[word] = embedding
    
    return embeddings

def find_similar_words(embeddings, word, top_k=5):
    """Find the top k most similar words using cosine similarity."""
    if word not in embeddings:
        print(f"Word '{word}' not found in embeddings")
        return
    
    # Get the target word's embedding
    target_embedding = embeddings[word].reshape(1, -1)
    
    # Calculate similarities with all other words
    similarities = {}
    for other_word, other_embedding in embeddings.items():
        if other_word != word:
            similarity = cosine_similarity(target_embedding, other_embedding.reshape(1, -1))[0][0]
            similarities[other_word] = similarity
    
    # Get top k similar words
    top_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Print results
    print(f"\nTop {top_k} most similar words to '{word}':")
    for i, (similar_word, similarity) in enumerate(top_words, 1):
        print(f"{i}. {similar_word}: {similarity:.4f}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    # Load text8 data
    text8_path = os.path.join('models', 'text8_embeddings', 'text8')
    print("Loading text8 data...")
    text8_words = load_text8_data(text8_path)
    
    # Get unique words
    unique_words = list(set(text8_words))
    print(f"Found {len(unique_words)} unique words")
    
    # Get BERT embeddings
    print("Getting BERT embeddings...")
    embeddings = get_bert_embeddings(unique_words, tokenizer, model, device)
    
    # Save embeddings
    output_path = os.path.join('models', 'text8_embeddings', 'inherited_bert_embeddings.pkl')
    print(f"Saving embeddings to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save vocabulary
    vocab_path = os.path.join('models', 'text8_embeddings', 'inherited_bert_vocab.json')
    print(f"Saving vocabulary to {vocab_path}...")
    with open(vocab_path, 'w') as f:
        json.dump(list(embeddings.keys()), f)
    
    print("Done!")
    
    # Interactive word similarity search
    print("\nWord Similarity Search")
    print("Enter a word to find similar words (or 'quit' to exit)")
    while True:
        word = input("\nEnter a word: ").strip().lower()
        if word == 'quit':
            break
        find_similar_words(embeddings, word)

if __name__ == "__main__":
    main()
