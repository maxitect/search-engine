import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import pickle
import src.config as config
from src.dataset import MSMARCOTripletDataset
from src.utils.model_loader import load_twotowers
from utils.models.twotowers import cosine_similarity

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_accuracy(query_emb, pos_doc_emb, neg_doc_emb):
    # Calculate similarity scores
    pos_sim = cosine_similarity(query_emb, pos_doc_emb)
    neg_sim = cosine_similarity(query_emb, neg_doc_emb)

    # Check if positive similarity is higher than negative similarity
    correct = (pos_sim > neg_sim).float()
    return correct.mean().item()


if __name__ == "__main__":
    print(f'Using device: {dev}')

    # Load vocabulary
    vocab_to_int = pickle.load(open(config.VOCAB_TO_ID_PATH, 'rb'))

    # Load pre-trained two-tower model
    model = load_twotowers(vocab_to_int)
    model.to(dev)
    model.eval()

    # Load validation data
    val_df = pd.read_parquet('ms_marco_validation.parquet')
    print(f'Validation size: {len(val_df)}')

    # Create validation dataset
    val_dataset = MSMARCOTripletDataset(
        df=val_df,
        max_query_len=config.MAX_QUERY_LEN,
        max_doc_len=config.MAX_DOC_LEN,
        max_samples=config.NUMBER_OF_SAMPLES
    )
    print(f'Validation triplet set size: {len(val_dataset)}')

    # Create validation dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.TWOTOWERS_BATCH_SIZE,
        shuffle=False
    )
    print(f'Validation batches: {len(val_dataloader)}')

    # Evaluate model
    total_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='Evaluating'):
            query_ids = batch['query_ids'].to(dev)
            pos_doc_ids = batch['pos_doc_ids'].to(dev)
            neg_doc_ids = batch['neg_doc_ids'].to(dev)

            # Forward pass
            query_emb = model.query_tower(query_ids)
            pos_doc_emb = model.doc_tower(pos_doc_ids)
            neg_doc_emb = model.doc_tower(neg_doc_ids)

            # Calculate accuracy
            batch_accuracy = calculate_accuracy(
                query_emb, pos_doc_emb, neg_doc_emb)
            total_accuracy += batch_accuracy

    # Calculate and print average accuracy
    avg_accuracy = total_accuracy / len(val_dataloader)
    print(f'Validation accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)')
