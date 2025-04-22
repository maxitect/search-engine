import datetime
import torch
import wandb
import pickle
import pandas as pd
from tqdm import tqdm
import os
import src.model as model
import src.config as config
from src.dataset import MSMARCODataset, generate_triplets
from src.evaluate import evaluate_progress

# Device configuration
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {dev}')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
print(f'Timestamp: {ts}')

# Load vocabulary
vocab_to_int = pickle.load(open(config.VOCAB_TO_ID_PATH, 'rb'))
print(f'Vocabulary size: {len(vocab_to_int)}')

# Load data from parquet files
train_df = pd.read_parquet('ms_marco_train.parquet')
val_df = pd.read_parquet('ms_marco_validation.parquet')
print(f'Train size: {len(train_df)}')
print(f'Validation size: {len(val_df)}')

# Create datasets
train_dataset = MSMARCODataset(
    queries=train_df['queries'].tolist(),
    documents=train_df['documents'].tolist(),
    labels=train_df['labels'].tolist(),
    vocab_to_int=vocab_to_int,
    max_query_len=config.MAX_QUERY_LEN,
    max_doc_len=config.MAX_DOC_LEN
)
print(f'Train dataset size: {len(train_dataset)}')

val_dataset = MSMARCODataset(
    queries=train_df['queries'].tolist(),
    documents=train_df['documents'].tolist(),
    labels=train_df['labels'].tolist(),
    vocab_to_int=vocab_to_int,
    max_query_len=config.MAX_QUERY_LEN,
    max_doc_len=config.MAX_DOC_LEN
)
print(f'Validation dataset size: {len(val_dataset)}')

# Generate triplets
train_triplets = generate_triplets(train_dataset, config.TWOTOWERS_BATCH_SIZE)
val_triplets = generate_triplets(val_dataset, config.TWOTOWERS_BATCH_SIZE)
print(f'Train triplets size: {len(train_triplets)}')
print(f'Validation triplets size: {len(val_triplets)}')

# Load pretrained SkipGram model
skipgram = model.SkipGram(len(vocab_to_int), config.EMBEDDING_DIM)
skipgram.load_state_dict(torch.load(config.SKIPGRAM_BEST_MODEL_PATH))
skipgram.to(dev)
skipgram.eval()
print(f'Loaded SkipGram model from {config.SKIPGRAM_BEST_MODEL_PATH}')
print(f'SkipGram model size: {sum(p.numel() for p in skipgram.parameters())}')

# Initialize query and document towers
qry_tower = model.QryTower(
    vocab_size=len(vocab_to_int),
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    output_dim=config.OUTPUT_DIM,
    skipgram_model=skipgram,
    freeze_embeddings=True
)
print(f'Query tower size: {sum(p.numel() for p in qry_tower.parameters())}')

doc_tower = model.DocTower(
    vocab_size=len(vocab_to_int),
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    output_dim=config.OUTPUT_DIM,
    skipgram_model=skipgram,
    freeze_embeddings=True
)
print(f'Document tower size: {sum(p.numel() for p in doc_tower.parameters())}')

# Create two-tower model
two_tower = model.TwoTowerModel(qry_tower, doc_tower)
two_tower.to(dev)
print(
    'Two-tower model size: '
    f'{sum(p.numel() for p in two_tower.parameters())}'
)

# Define optimizer
optimiser = torch.optim.Adam([
    {'params': qry_tower.parameters()},
    {'params': doc_tower.parameters()}
], lr=config.LEARNING_RATE)
print(f'Optimizer: {optimiser}')

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='min', factor=0.5, patience=3, verbose=True
)
print(f'Learning rate scheduler: {scheduler}')

# Initialize wandb
wandb.init(project='search-engine', name=f'two-tower_{ts}')

# Training loop
best_val_loss = float('inf')
print(f'Best validation loss: {best_val_loss:.4f}')

for epoch in range(config.EPOCHS):
    # Training
    two_tower.train()
    train_loss = 0
    print(f'\nEpoch {epoch+1}/{config.EPOCHS}')

    progress = tqdm(train_triplets, desc=f'Epoch {epoch+1} (Train)')
    for step, (query, pos_doc, neg_doc) in enumerate(progress):
        query = query.to(dev)
        pos_doc = pos_doc.to(dev)
        neg_doc = neg_doc.to(dev)

        optimiser.zero_grad()

        # Forward pass
        query_emb = qry_tower(query.unsqueeze(0))
        pos_doc_emb = doc_tower(pos_doc.unsqueeze(0))
        neg_doc_emb = doc_tower(neg_doc.unsqueeze(0))

        # Compute triplet loss
        loss = model.triplet_loss(
            query_emb, pos_doc_emb, neg_doc_emb, margin=config.MARGIN)

        # Backward pass and optimize
        loss.backward()
        optimiser.step()

        train_loss += loss.item()
        progress.set_postfix({'loss': loss.item()})

        if step % 10000 == 0:
            evaluate_progress(qry_tower, doc_tower, step + 1)

    avg_train_loss = train_loss / len(train_triplets)

    # Validation
    two_tower.eval()
    val_loss = 0

    with torch.no_grad():
        for query, pos_doc, neg_doc in tqdm(
            val_triplets,
            desc=f'Epoch {epoch+1} (Val)'
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
