from datasets import load_dataset
import random
from transformers import AutoTokenizer
import numpy as np

def load_msmarco_data():
    """Load MS MARCO dataset from HuggingFace"""
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    return dataset

def create_triples(dataset, num_negatives=1):
    """Generate (query, positive, negative) triples"""
    triples = []
    for example in dataset['train']:
        query = example['query']
        positives = example['passages']['is_selected']
        positives_text = [text for text, selected in zip(example['passages']['passage_text'], positives) if selected]
        negatives_text = [text for text, selected in zip(example['passages']['passage_text'], positives) if not selected]
        
        # If no negatives in this example, sample from other examples
        if not negatives_text:
            negatives_text = random.sample(dataset['train']['passages']['passage_text'], num_negatives)
        
        for pos in positives_text:
            for neg in random.sample(negatives_text, min(num_negatives, len(negatives_text))):
                triples.append((query, pos, neg))
    return triples

def tokenize_triples(triples, tokenizer_name='bert-base-uncased'):
    """Tokenize all text in the triples"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_triples = []
    for query, pos, neg in triples:
        tokenized_query = tokenizer(query, truncation=True, padding='max_length', max_length=64)
        tokenized_pos = tokenizer(pos, truncation=True, padding='max_length', max_length=256)
        tokenized_neg = tokenizer(neg, truncation=True, padding='max_length', max_length=256)
        tokenized_triples.append((tokenized_query, tokenized_pos, tokenized_neg))
    return tokenized_triples