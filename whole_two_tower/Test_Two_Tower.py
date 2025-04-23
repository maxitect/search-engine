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
        'hard_queries': [],
        'easy_queries': []
    }
    
    total_queries = 0
    query_scores = []
    
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
                
                # Handle single prediction case
                if predictions.dim() == 1:
                    pred_score = predictions[i].item()
                    is_correct = (pred_score > 0.5) == is_selected[i].item()
                    
                    # Update metrics for single prediction
                    metrics['mrr'] += 1.0 if is_correct else 0.0
                    for k in k_values:
                        metrics['precision'][k] += 1.0 if is_correct else 0.0
                        metrics['recall'][k] += 1.0 if is_correct else 0.0
                        metrics['ndcg'][k] += 1.0 if is_correct else 0.0
                    
                    # Track query performance
                    metrics['query_length_metrics'][query_length]['correct'] += 1 if is_correct else 0
                    metrics['query_length_metrics'][query_length]['total'] += 1
                    
                    # Track query scores
                    query_score = {
                        'query_length': query_length,
                        'score': pred_score,
                        'is_correct': is_correct
                    }
                    query_scores.append(query_score)
                else:
                    # Handle multiple predictions case
                    relevant_scores = predictions[i][relevant_mask].unsqueeze(0) if relevant_mask.any() else torch.tensor([0.0], device=device)
                    irrelevant_scores = predictions[i][~relevant_mask].unsqueeze(0) if (~relevant_mask).any() else torch.tensor([0.0], device=device)
                    
                    # Calculate MRR
                    if relevant_mask.any():
                        rank = (predictions[i] > relevant_scores[0]).sum().item() + 1
                        metrics['mrr'] += 1.0 / rank
                    
                    # Calculate Precision@K and Recall@K
                    for k in k_values:
                        top_k = torch.topk(predictions[i], min(k, len(predictions[i]))).indices
                        relevant_in_top_k = is_selected[i][top_k].sum().item()
                        total_relevant = is_selected[i].sum().item()
                        
                        metrics['precision'][k] += relevant_in_top_k / min(k, len(predictions[i]))
                        if total_relevant > 0:
                            metrics['recall'][k] += relevant_in_top_k / total_relevant
                    
                    # Calculate NDCG@K
                    for k in k_values:
                        ideal_sorted = torch.sort(is_selected[i], descending=True)[0]
                        actual_sorted = torch.sort(predictions[i], descending=True)[0]
                        
                        dcg = 0
                        idcg = 0
                        for j in range(min(k, len(predictions[i]))):
                            dcg += actual_sorted[j] / math.log2(j + 2)
                            idcg += ideal_sorted[j] / math.log2(j + 2)
                        
                        if idcg > 0:
                            metrics['ndcg'][k] += dcg / idcg
                    
                    # Calculate pairwise accuracy
                    if relevant_scores.numel() > 0 and irrelevant_scores.numel() > 0:
                        correct_pairs = (relevant_scores.unsqueeze(1) > irrelevant_scores.unsqueeze(0)).sum().item()
                        total_pairs = relevant_scores.numel() * irrelevant_scores.numel()
                        metrics['pairwise_accuracy'] += correct_pairs / total_pairs
                    
                    # Track query performance by length
                    if relevant_mask.any():
                        metrics['query_length_metrics'][query_length]['correct'] += 1
                    metrics['query_length_metrics'][query_length]['total'] += 1
                    
                    # Track query scores for hard/easy analysis
                    query_score = {
                        'query_length': query_length,
                        'score': predictions[i][relevant_mask].mean().item() if relevant_mask.any() else 0,
                        'is_correct': relevant_mask.any() and predictions[i][relevant_mask].mean() > predictions[i][~relevant_mask].mean()
                    }
                    query_scores.append(query_score)
    
    # Average the metrics
    metrics['mrr'] /= total_queries
    for k in k_values:
        metrics['precision'][k] /= total_queries
        metrics['recall'][k] /= total_queries
        metrics['ndcg'][k] /= total_queries
    metrics['pairwise_accuracy'] /= total_queries
    
    # Identify hard and easy queries
    query_scores.sort(key=lambda x: x['score'])
    metrics['hard_queries'] = query_scores[:10]  # 10 hardest queries
    metrics['easy_queries'] = query_scores[-10:]  # 10 easiest queries
    
    return metrics

def print_metrics(metrics):
    """Print the calculated metrics in a readable format."""
    print("\nEvaluation Metrics:")
    print("-" * 50)
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
    
    print("\nHardest Queries (lowest scores):")
    for query in metrics['hard_queries']:
        print(f"  Length {query['query_length']}: Score {query['score']:.4f}, Correct: {query['is_correct']}")
    
    print("\nEasiest Queries (highest scores):")
    for query in metrics['easy_queries']:
        print(f"  Length {query['query_length']}: Score {query['score']:.4f}, Correct: {query['is_correct']}")

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
