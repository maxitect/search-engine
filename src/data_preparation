# data_preparation.py
from datasets import load_dataset
from collections import defaultdict
import random

def load_msmarco():
    """Load and prepare MS MARCO dataset"""
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    return dataset

def generate_triples(dataset, neg_samples=1):
    """Generate query-positive-negative triples"""
    triples = []
    doc_pool = defaultdict(list)
    
    # Build document pool
    for sample in dataset['train']:
        doc_pool[sample['query']].append(sample['passages']['passage_text'][0])
    
    # Generate triples
    for query, pos_docs in doc_pool.items():
        for pos_doc in pos_docs:
            # Negative sampling
            neg_doc = random.choice(random.choice(list(doc_pool.values())))
            while neg_doc == pos_doc:
                neg_doc = random.choice(random.choice(list(doc_pool.values())))
            triples.append((query, pos_doc, neg_doc))
    
    return triples

def split_data(triples, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test"""
    random.shuffle(triples)
    train_size = int(len(triples) * train_ratio)
    val_size = int(len(triples) * val_ratio)
    
    return {
        'train': triples[:train_size],
        'val': triples[train_size:train_size+val_size],
        'test': triples[train_size+val_size:]
    }