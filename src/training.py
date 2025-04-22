# src/training.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, query, pos, neg):
        pos_dist = torch.sum((query - pos)**2, dim=1)
        neg_dist = torch.sum((query - neg)**2, dim=1)
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()

def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4, checkpoint_path='../models/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = TripletLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')  # Initialize with a large number

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            query_input = {k: v.to(device) for k, v in batch[0].items()}
            pos_input = {k: v.to(device) for k, v in batch[1].items()}
            neg_input = {k: v.to(device) for k, v in batch[2].items()}

            optimizer.zero_grad()

            # Forward pass
            query_repr, pos_repr = model(query_input, pos_input)
            _, neg_repr = model(query_input, neg_input)

            # Compute loss
            loss = criterion(query_repr, pos_repr, neg_repr)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                query_input = {k: v.to(device) for k, v in batch[0].items()}
                pos_input = {k: v.to(device) for k, v in batch[1].items()}
                neg_input = {k: v.to(device) for k, v in batch[2].items()}

                query_repr, pos_repr = model(query_input, pos_input)
                _, neg_repr = model(query_input, neg_input)

                val_loss += criterion(query_repr, pos_repr, neg_repr).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint if validation loss has improved
        if avg_val_loss < best_val_loss:
            print(f"✅ New best model found at epoch {epoch+1}, saving to {checkpoint_path}")
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)

    return model
def collate_batch(batch):
    """Collate function to merge tokenized triples"""
    query_batch = {}
    pos_batch = {}
    neg_batch = {}

    for query, pos, neg in batch:
        for k in query:
            query_batch.setdefault(k, []).append(query[k].squeeze(0))
            pos_batch.setdefault(k, []).append(pos[k].squeeze(0))
            neg_batch.setdefault(k, []).append(neg[k].squeeze(0))

    # Stack tensors
    for k in query_batch:
        query_batch[k] = torch.stack(query_batch[k])
        pos_batch[k] = torch.stack(pos_batch[k])
        neg_batch[k] = torch.stack(neg_batch[k])

    return query_batch, pos_batch, neg_batch
