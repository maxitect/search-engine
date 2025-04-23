import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from two_tower_model import TwoTowerModel
from customised_train_tower import CustomTwoTowerModel, CustomMSMARCODataset
import json
from tqdm import tqdm
from torch.serialization import add_safe_globals
import numpy.core.multiarray
from numpy import dtype
from numpy.dtypes import Float64DType
from collections import defaultdict
import math

# Add numpy types to safe globals
add_safe_globals([numpy.core.multiarray.scalar, dtype, Float64DType])

def load_gensim_model(model_path, gensim_model_path, hidden_dim):
    """Load the trained model with Gensim embeddings."""
    model = TwoTowerModel(gensim_model_path=gensim_model_path, hidden_dim=hidden_dim)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_custom_model(model_path, vocab_size, embedding_dim, hidden_dim):
    """Load the trained model with custom embeddings."""
    model = CustomTwoTowerModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def calculate_metrics(model, data_loader, device, k_values=[1, 5, 10]):
    """Calculate comprehensive ranking metrics."""
    model.eval()
    metrics = {
        'mrr': 0.0,
        'precision': {k: 0.0 for k in k_values},
        'recall': {k: 0.0 for k in k_values},
        'ndcg': {k: 0.0 for k in k_values},
        'pairwise_accuracy': 0.0,
        'query_length_metrics': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'query_performance': defaultdict(list)  # Store performance by query length
    }
    
    total_queries = 0
    queries_with_relevant = 0  # Track queries that have relevant passages
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating metrics"):
            query = batch['query'].to(device)
            passage = batch['passage'].to(device)
            is_selected = batch['is_selected'].to(device)
            
            # Get scores
            scores = model(query, passage)
            predictions = torch.sigmoid(scores)
            
            # Calculate metrics for each query in the batch
            batch_size = query.size(0)
            for i in range(batch_size):
                total_queries += 1
                query_length = (query[i] != 0).sum().item()
                
                # Get relevant and irrelevant passages
                relevant_mask = is_selected[i].bool()
                if not relevant_mask.any():
                    continue  # Skip queries with no relevant passages
                
                queries_with_relevant += 1
                
                # Ensure predictions and labels are 1D tensors
                pred_scores = predictions[i].view(-1)
                labels = is_selected[i].view(-1)
                
                # Sort predictions and get ranks
                sorted_indices = torch.argsort(pred_scores, descending=True)
                sorted_labels = labels[sorted_indices]
                
                # Calculate MRR
                relevant_indices = (sorted_labels == 1).nonzero()
                if relevant_indices.numel() > 0:
                    first_relevant = relevant_indices.min().item()
                    metrics['mrr'] += 1.0 / (first_relevant + 1)
                
                # Calculate Precision@K and Recall@K
                for k in k_values:
                    # Ensure k doesn't exceed the number of passages
                    k = min(k, len(sorted_labels))
                    top_k = sorted_labels[:k]
                    relevant_in_top_k = top_k.sum().item()
                    total_relevant = sorted_labels.sum().item()
                    
                    metrics['precision'][k] += relevant_in_top_k / k
                    if total_relevant > 0:
                        metrics['recall'][k] += relevant_in_top_k / total_relevant
                
                # Calculate NDCG@K
                for k in k_values:
                    # Ensure k doesn't exceed the number of passages
                    k = min(k, len(sorted_labels))
                    ideal_sorted = torch.sort(labels, descending=True)[0]
                    actual_sorted = sorted_labels
                    
                    dcg = 0
                    idcg = 0
                    for j in range(k):
                        dcg += actual_sorted[j] / math.log2(j + 2)
                        idcg += ideal_sorted[j] / math.log2(j + 2)
                    
                    if idcg > 0:
                        metrics['ndcg'][k] += dcg / idcg
                
                # Calculate pairwise accuracy
                relevant_scores = pred_scores[relevant_mask]
                irrelevant_scores = pred_scores[~relevant_mask]
                if relevant_scores.numel() > 0 and irrelevant_scores.numel() > 0:
                    correct_pairs = (relevant_scores.unsqueeze(1) > irrelevant_scores.unsqueeze(0)).sum().item()
                    total_pairs = relevant_scores.numel() * irrelevant_scores.numel()
                    metrics['pairwise_accuracy'] += correct_pairs / total_pairs
                
                # Track query performance
                is_correct = (relevant_scores.mean() > irrelevant_scores.mean()).item()
                metrics['query_length_metrics'][query_length]['correct'] += 1 if is_correct else 0
                metrics['query_length_metrics'][query_length]['total'] += 1
                
                # Store detailed performance
                metrics['query_performance'][query_length].append({
                    'correct': is_correct,
                    'score_diff': (relevant_scores.mean() - irrelevant_scores.mean()).item()
                })
    
    # Average the metrics only over queries with relevant passages
    if queries_with_relevant > 0:
        metrics['mrr'] /= queries_with_relevant
        for k in k_values:
            metrics['precision'][k] /= queries_with_relevant
            metrics['recall'][k] /= queries_with_relevant
            metrics['ndcg'][k] /= queries_with_relevant
        metrics['pairwise_accuracy'] /= queries_with_relevant
    
    # Calculate confidence metrics by query length
    confidence_metrics = {}
    for length, performances in metrics['query_performance'].items():
        if len(performances) > 10:  # Only consider lengths with sufficient samples
            avg_score_diff = np.mean([p['score_diff'] for p in performances])
            std_score_diff = np.std([p['score_diff'] for p in performances])
            confidence_metrics[length] = {
                'avg_score_diff': avg_score_diff,
                'std_score_diff': std_score_diff,
                'confidence': avg_score_diff / (std_score_diff + 1e-6)  # Signal-to-noise ratio
            }
    
    metrics['confidence_metrics'] = confidence_metrics
    metrics['total_queries'] = total_queries
    metrics['queries_with_relevant'] = queries_with_relevant
    return metrics

def print_metrics(metrics):
    """Print the calculated metrics in a readable format."""
    print("\nEvaluation Metrics:")
    print("-" * 50)
    print(f"Total Queries: {metrics['total_queries']}")
    print(f"Queries with Relevant Passages: {metrics['queries_with_relevant']}")
    print(f"Mean Reciprocal Rank (MRR): {metrics['mrr']:.4f}")
    print("\nPrecision@K:")
    for k, p in metrics['precision'].items():
        print(f"  P@{k}: {p:.4f}")
    
    print("\nRecall@K:")
    for k, r in metrics['recall'].items():
        print(f"  R@{k}: {r:.4f}")
    
    print("\nNDCG@K:")
    for k, n in metrics['ndcg'].items():
        print(f"  NDCG@{k}: {n:.4f}")
    
    print(f"\nPairwise Accuracy: {metrics['pairwise_accuracy']:.4f}")
    
    print("\nPerformance by Query Length:")
    for length, stats in sorted(metrics['query_length_metrics'].items()):
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  Length {length}: {accuracy:.4f} ({stats['correct']}/{stats['total']})")
    
    print("\nModel Confidence by Query Length:")
    print("(Higher values indicate more confident predictions)")
    for length, conf in sorted(metrics['confidence_metrics'].items()):
        print(f"  Length {length}:")
        print(f"    Average Score Difference: {conf['avg_score_diff']:.4f}")
        print(f"    Standard Deviation: {conf['std_score_diff']:.4f}")
        print(f"    Confidence Score: {conf['confidence']:.4f}")

def main():
    # Ask user which model to test
    print("Which model would you like to test?")
    print("1. Top_Tower_Epoch.pth (Gensim embeddings)")
    print("2. Custom_Top_Tower_Epoch.pth (Custom embeddings)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if choice == "1":
        # Configuration for Gensim model
        config = {
            'model_path': '/root/search-engine/models/Top_Tower_Epoch.pth',
            'gensim_model_path': '/root/search-engine/models/text8_embeddings/word2vec_model',
            'val_path': '/root/search-engine/data/msmarco/val.json',
            'hidden_dim': 256,
            'batch_size': 32
        }
        
        # Load model
        print("Loading Gensim-based model...")
        model = load_gensim_model(config['model_path'], config['gensim_model_path'], config['hidden_dim'])
        model = model.to(device)
        
        # Load validation dataset
        print("Loading validation dataset...")
        val_dataset = MSMARCODataset(config['val_path'])
        
    elif choice == "2":
        # Configuration for custom model
        config = {
            'model_path': '/root/search-engine/models/Custom_Top_Tower_Epoch.pth',
            'val_path': '/root/search-engine/data/msmarco/val.json',
            'vocab_size': 100000,
            'embedding_dim': 300,
            'hidden_dim': 256,
            'batch_size': 32
        }
        
        # Load model
        print("Loading custom embeddings model...")
        model = load_custom_model(
            config['model_path'],
            config['vocab_size'],
            config['embedding_dim'],
            config['hidden_dim']
        )
        model = model.to(device)
        
        # Load validation dataset
        print("Loading validation dataset...")
        val_dataset = CustomMSMARCODataset(config['val_path'])
        
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return
    
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Calculate comprehensive metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(model, val_loader, device)
    
    # Print metrics
    print_metrics(metrics)

if __name__ == '__main__':
    main()
