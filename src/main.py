# src/main.py
from data_preparation import load_msmarco_data, create_triples, tokenize_triples
from model import TwoTowerModel
from training import train_model
from torch.utils.data import DataLoader, TensorDataset
import torch

def create_dataloader(query_input, pos_input, neg_input, batch_size=32):
    """Create a DataLoader from tokenized inputs"""
    dataset = TensorDataset(
        query_input['input_ids'],
        query_input['attention_mask'],
        pos_input['input_ids'],
        pos_input['attention_mask'],
        neg_input['input_ids'],
        neg_input['attention_mask']
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    # Load and prepare data
    print("Loading data...")
    dataset = load_msmarco_data()
    triples = create_triples(dataset[:1000], num_negatives=1)  # Use smaller subset for demo
    query_input, pos_input, neg_input = tokenize_triples(triples)
    
    # Split into train/val
    split_idx = int(0.8 * len(triples))
    
    # Create DataLoaders
    train_loader = create_dataloader(
        {k: v[:split_idx] for k, v in query_input.items()},
        {k: v[:split_idx] for k, v in pos_input.items()},
        {k: v[:split_idx] for k, v in neg_input.items()},
        batch_size=32
    )
    
    val_loader = create_dataloader(
        {k: v[split_idx:] for k, v in query_input.items()},
        {k: v[split_idx:] for k, v in pos_input.items()},
        {k: v[split_idx:] for k, v in neg_input.items()},
        batch_size=32
    )
    
    # Initialize model
    model = TwoTowerModel()
    
    # Train
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, num_epochs=3)
    
    # Save model
    torch.save(trained_model.state_dict(), '../models/two_tower_model.pth')
    print("Model saved!")

if __name__ == "__main__":
    main()