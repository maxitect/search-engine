import json
import os
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from datasets import load_dataset
from multiprocessing import Pool, cpu_count
import random

def load_gensim_model(model_path):
    """Load the Gensim model and return vocabulary."""
    print("Loading Gensim model...")
    model = Word2Vec.load(model_path)
    vocab = {word: idx for idx, word in enumerate(model.wv.index_to_key)}
    print(f"Loaded vocabulary with {len(vocab)} words")
    return model, vocab

def process_text_batch(texts, vocab, max_length):
    """Process a batch of texts into word indices."""
    processed = []
    for text in texts:
        words = text.lower().split()
        indices = [vocab.get(word, 0) for word in words[:max_length]]
        indices.extend([0] * (max_length - len(indices)))
        processed.append(indices)
    return processed

def process_example_batch(examples, vocab):
    """Process a batch of examples."""
    batch_data = []
    for example in examples:
        query = example['query']
        passages = example['passages']
        
        passage_texts = passages['passage_text']
        is_selected = passages['is_selected']
        
        # Find positive and negative passages
        positive_indices = [i for i, selected in enumerate(is_selected) if selected]
        negative_indices = [i for i, selected in enumerate(is_selected) if not selected]
        
        # Skip if no positive passages
        if not positive_indices:
            continue
            
        # Randomly select one positive passage
        pos_idx = random.choice(positive_indices)
        
        # Select 5 negative passages (or all if less than 5)
        if len(negative_indices) >= 5:
            neg_indices = random.sample(negative_indices, 5)
        else:
            neg_indices = negative_indices
        
        # Combine selected passages
        selected_indices = [pos_idx] + neg_indices
        selected_texts = [passage_texts[i] for i in selected_indices]
        selected_labels = [1] + [0] * len(neg_indices)
        
        # Process query and passages
        query_indices = process_text_batch([query], vocab, 32)[0]
        passage_indices = process_text_batch(selected_texts, vocab, 256)
        
        batch_data.append({
            'query': query_indices,
            'passages': [{'passage_text': indices} for indices in passage_indices],
            'is_selected': selected_labels
        })
    
    return batch_data

def process_batch_wrapper(args):
    """Wrapper function for multiprocessing."""
    batch, vocab = args
    return process_example_batch(batch, vocab)

def prepare_dataset():
    """Prepare the MS MARCO dataset for training using personalized Gensim model."""
    # Load your personalized Gensim model
    print("Loading personalized Gensim model...")
    model, vocab = load_gensim_model('/root/search-engine/models/text8_embeddings/word2vec_model')
    print(f"Loaded vocabulary with {len(vocab)} words")
    
    # Create output directories
    os.makedirs('/root/search-engine/data/msmarco', exist_ok=True)
    
    # Load MS MARCO dataset from Hugging Face
    print("Loading MS MARCO dataset from Hugging Face...")
    dataset = load_dataset("ms_marco", "v1.1")
    
    # Process data in batches
    batch_size = 1000
    num_workers = min(cpu_count(), 4)  # Use up to 4 CPU cores
    
    # Process training data
    print("Processing training data...")
    train_data = []
    train_examples = list(dataset['train'])
    
    # Process in parallel batches
    with Pool(num_workers) as pool:
        batches = [(train_examples[i:i + batch_size], vocab) 
                  for i in range(0, len(train_examples), batch_size)]
        for batch_results in tqdm(pool.imap_unordered(
            process_batch_wrapper, 
            batches
        ), total=len(batches), desc="Processing batches"):
            train_data.extend(batch_results)
    
    print(f"\nProcessed {len(train_data)} training examples")
    
    # Process validation data
    print("\nProcessing validation data...")
    val_data = []
    val_examples = list(dataset['validation'])
    
    # Process in parallel batches
    with Pool(num_workers) as pool:
        batches = [(val_examples[i:i + batch_size], vocab) 
                  for i in range(0, len(val_examples), batch_size)]
        for batch_results in tqdm(pool.imap_unordered(
            process_batch_wrapper, 
            batches
        ), total=len(batches), desc="Processing batches"):
            val_data.extend(batch_results)
    
    print(f"Processed {len(val_data)} validation examples")
    
    # Process test data
    print("\nProcessing test data...")
    test_data = []
    test_examples = list(dataset['test'])
    
    # Process in parallel batches
    with Pool(num_workers) as pool:
        batches = [(test_examples[i:i + batch_size], vocab) 
                  for i in range(0, len(test_examples), batch_size)]
        for batch_results in tqdm(pool.imap_unordered(
            process_batch_wrapper, 
            batches
        ), total=len(batches), desc="Processing batches"):
            test_data.extend(batch_results)
    
    print(f"Processed {len(test_data)} test examples")
    
    # Save data in JSON format
    print(f"\nSaving train set ({len(train_data)} examples)...")
    with open('/root/search-engine/data/msmarco/train.json', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saving validation set ({len(val_data)} examples)...")
    with open('/root/search-engine/data/msmarco/val.json', 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saving test set ({len(test_data)} examples)...")
    with open('/root/search-engine/data/msmarco/test.json', 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print("Dataset preparation complete!")

if __name__ == '__main__':
    prepare_dataset() 