from datasets import load_dataset
import random
from transformers import AutoTokenizer
from collections import defaultdict

def load_msmarco_data():
    """Load MS MARCO dataset from HuggingFace"""
    dataset = load_dataset("ms_marco", "v1.1")
    return dataset

def create_triples(dataset, num_negatives=1):
    """Generate (query, positive, negative) triples"""
    triples = []
    
    # First collect all passages for negative sampling
    all_passages = []
    passage_id_map = {}
    
    # Build a map of query to relevant passages
    query_positives = defaultdict(list)
    
    for example in dataset['train']:
        query = example['query']
        passages = example['passages']
        is_selected = example['passages']['is_selected']
        
        # Store positive passages for this query
        for passage, selected in zip(passages['passage_text'], is_selected):
            if selected:
                query_positives[query].append(passage)
            all_passages.append(passage)
    
    # Now create triples
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
    """Tokenize all text in the triples"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_triples = []
    
    for query, pos, neg in triples:
        tokenized_query = tokenizer(
            query, 
            truncation=True, 
            padding='max_length', 
            max_length=64,
            return_tensors='pt'
        )
        tokenized_pos = tokenizer(
            pos, 
            truncation=True, 
            padding='max_length', 
            max_length=256,
            return_tensors='pt'
        )
        tokenized_neg = tokenizer(
            neg, 
            truncation=True, 
            padding='max_length', 
            max_length=256,
            return_tensors='pt'
        )
        tokenized_triples.append((tokenized_query, tokenized_pos, tokenized_neg))
    
    return tokenized_triples