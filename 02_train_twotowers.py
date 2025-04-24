import argparse
import datetime
import torch
from torch.utils.data import DataLoader
import wandb
import pickle
import pandas as pd
from tqdm import tqdm
import os
import src.models.twotowers as model
import src.config as config
from src.dataset import MSMARCOTripletDataset
from src.evaluate import evaluate_progress
from src.utils.model_loader import download_model, load_skipgram

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download SkipGram model')
    parser.add_argument(
        '--download_sg',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    args = parser.parse_args()
    # Download model if required
    download_model(
        args.download_sg, "skipgram-best", config.SKIPGRAM_CHECKPOINT_DIR
    )
    # Device configuration
    print(f'Using device: {dev}')
    ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    # Load vocabulary and skipgram model
    vocab_to_int = pickle.load(open(config.VOCAB_TO_ID_PATH, 'rb'))
    skipgram = load_skipgram(vocab_to_int)

    # Load data from parquet files
    train_df = pd.read_parquet('ms_marco_train.parquet')
    val_df = pd.read_parquet('ms_marco_validation.parquet')
    print(f'Train size: {len(train_df)}')
    print(f'Validation size: {len(val_df)}')

    # Create datasets using our improved MSMARCOTripletDataset
    train_dataset = MSMARCOTripletDataset(
        df=train_df,
        max_query_len=config.MAX_QUERY_LEN,
        max_doc_len=config.MAX_DOC_LEN,
        max_samples=config.NUMBER_OF_SAMPLES
    )
    print(f'Train triplet set size: {len(train_dataset)}')

    val_dataset = MSMARCOTripletDataset(
        df=val_df,
        max_query_len=config.MAX_QUERY_LEN,
        max_doc_len=config.MAX_DOC_LEN,
        max_samples=config.NUMBER_OF_SAMPLES
    )
    print(f'Validation triplet set size: {len(val_dataset)}')

    # Generate triplets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TWOTOWERS_BATCH_SIZE,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.TWOTOWERS_BATCH_SIZE,
        shuffle=True
    )
    print(f'Train batch size: {len(train_dataloader)}')
    print(f'Validation batch size: {len(val_dataloader)}')

    # Initialize query and document towers
    qry_tower = model.QryTower(
        vocab_size=len(vocab_to_int),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout_rate=config.DROPOUT_RATE,
        skipgram_model=skipgram,
        freeze_embeddings=True
    )
    print(
        'Query tower parameters: '
        f'{sum(p.numel() for p in qry_tower.parameters())}'
    )

    doc_tower = model.DocTower(
        vocab_size=len(vocab_to_int),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout_rate=config.DROPOUT_RATE,
        skipgram_model=skipgram,
        freeze_embeddings=True
    )
    print(
        'Document tower parameters: '
        f'{sum(p.numel() for p in doc_tower.parameters())}'
    )

    # Create two-tower model
    two_towers = model.TwoTowerModel(qry_tower, doc_tower)
    two_towers.to(dev)
    print(
        'Two-towers parameters: '
        f'{sum(p.numel() for p in two_towers.parameters())}'
    )

    optimiser = torch.optim.Adam(
        list(qry_tower.parameters()) + list(doc_tower.parameters()),
        lr=config.TWOTOWERS_LR
    )

    # Learning rate scheduler - remove verbose flag
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=3
    )

    # Initialize wandb
    wandb.init(project='search-engine', name=f'two-tower_{ts}')

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.TWOTOWERS_EPOCHS):
        # Training
        two_towers.train()
        train_loss = 0
        print(f'\nEpoch {epoch+1}/{config.TWOTOWERS_EPOCHS}')

        progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1} (Train)')
        for step, batch in enumerate(progress):
            query_ids = batch['query_ids'].to(dev)
            pos_doc_ids = batch['pos_doc_ids'].to(dev)
            neg_doc_ids = batch['neg_doc_ids'].to(dev)

            optimiser.zero_grad()

            # Forward pass
            query_emb = qry_tower(query_ids)
            pos_doc_emb = doc_tower(pos_doc_ids)
            neg_doc_emb = doc_tower(neg_doc_ids)

            # Compute triplet loss
            loss = model.triplet_loss_function(
                query_emb, pos_doc_emb, neg_doc_emb, margin=config.MARGIN)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                two_towers.parameters(), max_norm=1.0
            )
            optimiser.step()

            train_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})

            if step % 500 == 0:
                wandb.log({'test_step_loss': evaluate_progress(
                    qry_tower,
                    doc_tower,
                    f'step {step + 1}'
                )})

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation
        two_towers.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in tqdm(
                val_dataloader,
                desc=f'Epoch {epoch+1} (Val)'
            ):
                query_ids = batch['query_ids'].to(dev)
                pos_doc_ids = batch['pos_doc_ids'].to(dev)
                neg_doc_ids = batch['neg_doc_ids'].to(dev)

                # Forward pass
                query_emb = qry_tower(query_ids)
                pos_doc_emb = doc_tower(pos_doc_ids)
                neg_doc_emb = doc_tower(neg_doc_ids)

                # Compute triplet loss
                loss = model.triplet_loss_function(
                    query_emb,
                    pos_doc_emb,
                    neg_doc_emb,
                    margin=config.MARGIN
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        print(
            f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, '
            f'Val Loss = {avg_val_loss:.4f}')

        # Log to wandb
        wandb.log({
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'test_loss': evaluate_progress(
                qry_tower, doc_tower, f'epoch {epoch}'
            ),
            'epoch': epoch
        })

        # Save checkpoint
        checkpoint_path = os.path.join(
            config.TWOTOWERS_CHECKPOINT_DIR, f'two_towers_epoch_{epoch+1}.pth')
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
            }, config.TWOTOWERS_BEST_MODEL_PATH)
            # Create a special artifact for the best model
            best_artifact = wandb.Artifact('two-towers-best', type='model')
            best_artifact.add_file(config.TWOTOWERS_BEST_MODEL_PATH)
            wandb.log_artifact(best_artifact)
            patience_counter = 0
            print(
                f'New best model saved at epoch {epoch+1} '
                f'with validation loss {best_val_loss:.4f}'
            )
        else:
            patience_counter += 1

        artifact = wandb.Artifact('two-towers', type='model')
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

        if patience_counter >= config.TWOTOWERS_PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    print(f'Training completed! Best validation loss: {best_val_loss:.4f}')
    wandb.finish()
