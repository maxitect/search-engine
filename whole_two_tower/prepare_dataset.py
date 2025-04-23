import json
import os
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec

def load_gensim_model(model_path):
    """Load the Gensim model and return vocabulary."""
    print("Loading Gensim model...")
    model = Word2Vec.load(model_path)
    vocab = list(model.wv.index_to_key)
    print(f"Loaded vocabulary with {len(vocab)} words")
    return model, vocab

def word_to_index(word, vocab):
    """Convert a word to its index in the vocabulary."""
    return vocab.index(word) if word in vocab else 0  # 0 for OOV words

def process_text(text, vocab, max_length):
    """Process text into word indices with padding."""
    words = text.lower().split()
    indices = [word_to_index(word, vocab) for word in words[:max_length]]
    # Pad with zeros
    indices.extend([0] * (max_length - len(indices)))
    return indices

def prepare_dataset():
    """Prepare the MS MARCO dataset for training."""
    # Load Gensim model
    model, vocab = load_gensim_model('/root/search-engine/models/text8_embeddings/word2vec_model')
    
    # Create output directories
    os.makedirs('/root/search-engine/data/msmarco', exist_ok=True)
    
    # Load MS MARCO data
    print("Loading MS MARCO data...")
    with open('/root/search-engine/models/text8_embeddings/ms_marco_train.json', 'r') as f:
        passages = json.load(f)
    
    # Process data
    print("Processing data...")
    processed_data = []
    error_count = 0
    
    # Process each passage
    for passage in tqdm(passages, desc="Processing passages"):
        try:
            # Process the passage text
            passage_indices = process_text(passage, vocab, max_length=256)
            
            # Create a dummy query (you'll need to replace this with actual queries)
            dummy_query = "What is this passage about?"
            query_indices = process_text(dummy_query, vocab, max_length=32)
            
            # Create a processed item
            processed_data.append({
                'query': query_indices,
                'passages': [{
                    'passage_text': passage_indices,
                    'is_selected': 1  # Mark as selected since we're using single passages
                }]
            })
        except Exception as e:
            error_count += 1
            if error_count <= 3:  # Only show first 3 errors
                print(f"\nError processing passage: {e}")
                print(f"Problematic passage: {passage[:200]}...")
            continue
    
    if error_count > 0:
        print(f"\nSkipped {error_count} passages due to processing errors")
    
    # Split into train and validation
    train_size = int(len(processed_data) * 0.9)
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:]
    
    # Save train set
    print(f"\nSaving train set ({len(train_data)} examples)...")
    with open('/root/search-engine/data/msmarco/train.json', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Save validation set
    print(f"Saving validation set ({len(val_data)} examples)...")
    with open('/root/search-engine/data/msmarco/val.json', 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print("Dataset preparation complete!")

if __name__ == '__main__':
    prepare_dataset() 