import datetime
import requests
import torch
import wandb
import pickle
from tqdm import tqdm
import os
from dataset import MSMARCODataset, generate_triplets
import src.model as model
import src.config as config
from torch.utils.data import random_split

# Device configuration
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# Load vocabulary and dataset
vocab_to_int = pickle.load(open(config.VOCAB_TO_ID_PATH, 'rb'))
int_to_vocab = pickle.load(open(config.ID_TO_VOCAB_PATH, 'rb'))

# Download MS MARCO dataset
print("Downloading MS MARCO dataset...")
r = requests.get(
    "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/train-00000-of-00001.parquet"
)
with open("ms_marco_train.parquet", "wb") as f:
    f.write(r.content)


queries = []
documents = []
labels = []
# Load your actual data here
# ...

# Create dataset
dataset = MSMARCODataset(queries, documents, labels, vocab_to_int)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Generate triplets
train_triplets = generate_triplets(train_dataset, config.BATCH_SIZE)
val_triplets = generate_triplets(val_dataset, config.BATCH_SIZE)

# Load pretrained SkipGram model
skipgram = model.SkipGram(len(vocab_to_int), config.EMBEDDING_DIM)
skipgram.load_state_dict(torch.load(config.SKIPGRAM_BEST_MODEL_PATH))
skipgram.to(dev)
skipgram.eval()

# Initialize query and document towers
qry_tower = model.QryTower(
    vocab_size=len(vocab_to_int),
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    output_dim=config.OUTPUT_DIM,
    skipgram_model=skipgram,
    freeze_embeddings=True
)

doc_tower = model.DocTower(
    vocab_size=len(vocab_to_int),
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    output_dim=config.OUTPUT_DIM,
    skipgram_model=skipgram,
    freeze_embeddings=True
)

# Create two-tower model
two_tower = model.TwoTowerModel(qry_tower, doc_tower)
two_tower.to(dev)

# Define optimizer
optimizer = torch.optim.Adam([
    {'params': qry_tower.parameters()},
    {'params': doc_tower.parameters()}
], lr=config.LEARNING_RATE)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# Initialize wandb
wandb.init(project='msmarco-two-tower', name=f'{ts}')

# Training loop
best_val_loss = float('inf')

for epoch in range(config.EPOCHS):
    # Training
    two_tower.train()
    train_loss = 0

    progress = tqdm(train_triplets, desc=f'Epoch {epoch+1} (Train)')
    for query, pos_doc, neg_doc in progress:
        query = query.to(dev)
        pos_doc = pos_doc.to(dev)
        neg_doc = neg_doc.to(dev)

        optimizer.zero_grad()

        # Forward pass
        query_emb = qry_tower(query.unsqueeze(0))
        pos_doc_emb = doc_tower(pos_doc.unsqueeze(0))
        neg_doc_emb = doc_tower(neg_doc.unsqueeze(0))

        # Compute triplet loss
        loss = model.triplet_loss(
            query_emb, pos_doc_emb, neg_doc_emb, margin=config.MARGIN)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})

    avg_train_loss = train_loss / len(train_triplets)

    # Validation
    two_tower.eval()
    val_loss = 0

    with torch.no_grad():
        for query, pos_doc, neg_doc in tqdm(
            val_triplets, desc=f'Epoch {epoch+1} (Val)'
        ):
            query = query.to(dev)
            pos_doc = pos_doc.to(dev)
            neg_doc = neg_doc.to(dev)

            # Forward pass
            query_emb = qry_tower(query.unsqueeze(0))
            pos_doc_emb = doc_tower(pos_doc.unsqueeze(0))
            neg_doc_emb = doc_tower(neg_doc.unsqueeze(0))

            # Compute triplet loss
            loss = model.triplet_loss(
                query_emb, pos_doc_emb, neg_doc_emb, margin=config.MARGIN)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_triplets)

    # Log to wandb
    wandb.log({
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'epoch': epoch
    })

    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    print(
        f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, '
        f'Val Loss = {avg_val_loss:.4f}')

    # Save checkpoint
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR, f'two_tower_epoch_{epoch+1}.pth')
    torch.save({
        'query_tower': qry_tower.state_dict(),
        'doc_tower': doc_tower.state_dict(),
        'epoch': epoch,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
    }, checkpoint_path)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'query_tower': qry_tower.state_dict(),
            'doc_tower': doc_tower.state_dict(),
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, config.BEST_MODEL_PATH)
        print(f'New best model saved with validation loss {best_val_loss:.4f}')

print(f'Training completed! Best validation loss: {best_val_loss:.4f}')
wandb.finish()
