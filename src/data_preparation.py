# src/data_preparation.py
from datasets import load_dataset
import random
from transformers import AutoTokenizer
from collections import defaultdict
import torch

def load_msmarco_data():
    """Load MS MARCO dataset from HuggingFace"""
    dataset = load_dataset("ms_marco", "v1.1")
    return dataset

def create_triples(dataset, num_negatives=1):
    """Generate (query, positive, negative) triples"""
    triples = []
    
    # First collect all passages and build query-positive map
    all_passages = []
    query_positives = defaultdict(list)
    
    for example in dataset['train']:
        query = example['query']
        passages = example['passages']['passage_text']
        is_selected = example['passages']['is_selected']
        
        # Store positive passages for this query
        positives = [passages[i] for i, selected in enumerate(is_selected) if selected]
        query_positives[query].extend(positives)
        all_passages.extend(passages)
    
    # Create triples ensuring negatives aren't positives for the query
    for query, positives in query_positives.items():
        for pos in positives:
            # Sample negatives that aren't positives for this query
            negatives = []
            attempts = 0
            while len(negatives) < num_negatives and attempts < 100:
                candidate = random.choice(all_passages)
                if candidate not in positives:
                    negatives.append(candidate)
                attempts += 1
            
            if negatives:
                triples.append((query, pos, negatives[0]))
    
    return triples

def tokenize_triples(triples, tokenizer_name='bert-base-uncased'):
    """Tokenize all text in the triples with proper tensor formatting"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Initialize lists to store tokenized inputs
    query_inputs = []
    pos_inputs = []
    neg_inputs = []
    
    for query, pos, neg in triples:
        # Tokenize each component
        query_tokens = tokenizer(
            query, 
            truncation=True, 
            padding='max_length', 
            max_length=64,
            return_tensors='pt'
        )
        pos_tokens = tokenizer(
            pos, 
            truncation=True, 
            padding='max_length', 
            max_length=256,
            return_tensors='pt'
        )
        neg_tokens = tokenizer(
            neg, 
            truncation=True, 
            padding='max_length', 
            max_length=256,
            return_tensors='pt'
        )
        
        query_inputs.append(query_tokens)
        pos_inputs.append(pos_tokens)
        neg_inputs.append(neg_tokens)
    
    # Combine all batches
    def collate_batch(batch):
        return {
            'input_ids': torch.cat([item['input_ids'] for item in batch], dim=0),
            'attention_mask': torch.cat([item['attention_mask'] for item in batch], dim=0),
            'token_type_ids': torch.cat([item['token_type_ids'] for item in batch], dim=0) if 'token_type_ids' in batch[0] else None
        }
    
    return collate_batch(query_inputs), collate_batch(pos_inputs), collate_batch(neg_inputs)