from src.data_preparation import load_msmarco_data, create_triples, tokenize_triples
from src.model import TwoTowerModel
from src.training import train_model, collate_batch
from torch.utils.data import DataLoader, TensorDataset
import torch
import os

def main():
    # Load and prepare data
    print("Loading data...")
    dataset = load_msmarco_data()
    triples = create_triples(dataset, num_negatives=1)
    tokenized_triples = tokenize_triples(triples[:10000])  # Use smaller subset for demo
    
    # Prepare DataLoaders
    query_inputs = [t[0] for t in tokenized_triples]
    pos_inputs = [t[1] for t in tokenized_triples]
    neg_inputs = [t[2] for t in tokenized_triples]
    
    # Split into train/val
    split_idx = int(0.8 * len(tokenized_triples))
    train_data = list(zip(query_inputs[:split_idx], pos_inputs[:split_idx], neg_inputs[:split_idx]))
    val_data = list(zip(query_inputs[split_idx:], pos_inputs[split_idx:], neg_inputs[split_idx:]))
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_batch)

    
    # Initialize model
    model = TwoTowerModel()
    
    # Train
    print("Training model...")
    trained_model = train_model(model, train_loader, val_loader, num_epochs=3)
    
    # Save models
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save two_tower_model
    two_tower_path = os.path.join(models_dir, 'two_tower_model.pth')
    torch.save(trained_model.state_dict(), two_tower_path)
    
    # Save best_model
    best_model_path = os.path.join(models_dir, 'best_model.pth')
    torch.save(trained_model.state_dict(), best_model_path)
   
    print(f"Models saved in {models_dir}!")

if __name__ == "__main__":
    main() 