from src.inference import SearchEngine
from src.data_preparation import load_msmarco_data
import os
import numpy as np
from tqdm import tqdm
import json

def load_relevant_docs():
    """Load the relevant documents for each query from MS MARCO"""
    dataset = load_msmarco_data()
    relevant_docs = {}
    
    # Process the validation set to get ground truth
    for example in dataset['validation']:
        query = example['query']
        passages = example['passages']['passage_text']
        is_selected = example['passages']['is_selected']
        
        # Store relevant passages for this query
        relevant_passages = [passage for passage, selected in zip(passages, is_selected) if selected]
        if relevant_passages:
            relevant_docs[query] = relevant_passages
    
    return relevant_docs

def evaluate_search_engine(search_engine, test_queries, relevant_docs, top_k=5):
    """Evaluate the search engine's performance"""
    metrics = {
        'precision@k': [],
        'recall@k': [],
        'f1@k': [],
        'average_precision': []
    }
    
    for query in tqdm(test_queries, desc="Evaluating queries"):
        if query not in relevant_docs:
            continue
            
        # Get search results
        results = search_engine.search(query, top_k=top_k)
        retrieved_docs = [doc_id for doc_id, _ in results]
        
        # Get ground truth
        relevant_doc_ids = set()
        for passage in relevant_docs[query]:
            # Find the document ID for this passage
            for doc_id, doc_text in zip(search_engine.doc_ids, search_engine.documents):
                if doc_text == passage:
                    relevant_doc_ids.add(doc_id)
                    break
        
        # Calculate metrics
        relevant_retrieved = len(set(retrieved_docs) & relevant_doc_ids)
        
        # Precision@k
        precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
        metrics['precision@k'].append(precision)
        
        # Recall@k
        recall = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0
        metrics['recall@k'].append(recall)
        
        # F1@k
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics['f1@k'].append(f1)
        
        # Average Precision
        ap = 0
        relevant_count = 0
        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_doc_ids:
                relevant_count += 1
                ap += relevant_count / i
        ap = ap / len(relevant_doc_ids) if relevant_doc_ids else 0
        metrics['average_precision'].append(ap)
    
    # Calculate average metrics
    avg_metrics = {
        'precision@k': np.mean(metrics['precision@k']),
        'recall@k': np.mean(metrics['recall@k']),
        'f1@k': np.mean(metrics['f1@k']),
        'mean_average_precision': np.mean(metrics['average_precision'])
    }
    
    return avg_metrics

def main():
    # Initialize the search engine
    model_path = os.path.join('models', 'best_model.pth')
    search_engine = SearchEngine(model_path)
    
    # Load the document cache
    cache_path = os.path.join('models', 'document_cache.pth')
    search_engine.load_cache(cache_path)
    
    # Load relevant documents
    print("Loading relevant documents...")
    relevant_docs = load_relevant_docs()
    
    # Select test queries (using the first 100 queries from the validation set)
    test_queries = list(relevant_docs.keys())[:100]
    
    # Evaluate the search engine
    print("\nEvaluating search engine...")
    metrics = evaluate_search_engine(search_engine, test_queries, relevant_docs)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Precision@{5}: {metrics['precision@k']:.4f}")
    print(f"Recall@{5}: {metrics['recall@k']:.4f}")
    print(f"F1@{5}: {metrics['f1@k']:.4f}")
    print(f"Mean Average Precision: {metrics['mean_average_precision']:.4f}")
    
    # Save results to file
    results_path = os.path.join('models', 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main() 