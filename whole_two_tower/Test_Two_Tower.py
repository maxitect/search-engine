import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from two_tower_model import TwoTowerModel
from train_two_tower import MSMARCODataset
import json
from tqdm import tqdm
from torch.serialization import add_safe_globals
import numpy.core.multiarray
from numpy import dtype
from numpy.dtypes import Float64DType

# Add numpy types to safe globals
add_safe_globals([numpy.core.multiarray.scalar, dtype, Float64DType])

def load_model(model_path, gensim_model_path, hidden_dim):
    """Load the trained model."""
    model = TwoTowerModel(gensim_model_path=gensim_model_path, hidden_dim=hidden_dim)
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def calculate_mae(model, data_loader, device):
    """Calculate Mean Absolute Error."""
    model.eval()
    total_error = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating MAE"):
            query = batch['query'].to(device)
            passage = batch['passage'].to(device)
            is_selected = batch['is_selected'].to(device)
            
            scores = model(query, passage)
            predictions = torch.sigmoid(scores)
            
            error = torch.abs(predictions - is_selected)
            total_error += error.sum().item()
            total_samples += len(is_selected)
    
    return total_error / total_samples

def print_examples(model, dataset, device, num_examples=10):
    """Print example predictions."""
    model.eval()
    print("\nExample Predictions:")
    print("-" * 100)
    
    for i in range(min(num_examples, len(dataset))):
        item = dataset[i]
        query = item['query'].unsqueeze(0).to(device)
        passage = item['passage'].unsqueeze(0).to(device)
        is_selected = item['is_selected'].item()
        
        with torch.no_grad():
            score = model(query, passage)
            prediction = torch.sigmoid(score).item()
        
        print(f"Example {i+1}:")
        print(f"Ground Truth: {is_selected:.4f}")
        print(f"Prediction: {prediction:.4f}")
        print(f"Absolute Error: {abs(prediction - is_selected):.4f}")
        print("-" * 100)

def main():
    # Configuration
    config = {
        'model_path': '/root/search-engine/models/Top_Tower_Epoch.pth',
        'gensim_model_path': '/root/search-engine/models/text8_embeddings/word2vec_model',
        'val_path': '/root/search-engine/data/msmarco/val.json',
        'hidden_dim': 256,
        'batch_size': 32
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(config['model_path'], config['gensim_model_path'], config['hidden_dim'])
    model = model.to(device)
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = MSMARCODataset(config['val_path'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Calculate MAE
    print("\nCalculating MAE...")
    mae = calculate_mae(model, val_loader, device)
    print(f"\nMean Absolute Error: {mae:.4f}")
    
    # Print examples
    print_examples(model, val_dataset, device)

if __name__ == '__main__':
    main()
