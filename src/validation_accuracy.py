import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
from tqdm import tqdm
import config
import models.twotowers as model
from dataset import MSMARCOTripletDataset
from torch.utils.data import DataLoader

# Setup device
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {dev}')

# Load BERT
bert_model = BertModel.from_pretrained('bert-base-uncased').to(dev)

# Initialize towers
qry_tower = model.QryTower(
    vocab_size=config.VOCAB_SIZE,
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    output_dim=config.OUTPUT_DIM,
    dropout_rate=config.DROPOUT_RATE,
    bert_model=bert_model,
    freeze_embeddings=True
).to(dev)

doc_tower = model.DocTower(
    vocab_size=config.VOCAB_SIZE,
    embedding_dim=config.EMBEDDING_DIM,
    hidden_dim=config.HIDDEN_DIM,
    output_dim=config.OUTPUT_DIM,
    dropout_rate=config.DROPOUT_RATE,
    bert_model=bert_model,
    freeze_embeddings=True
).to(dev)

# Load best checkpoint
print("Loading best model checkpoint...")
checkpoint = torch.load(config.TWOTOWERS_BEST_MODEL_PATH)
qry_tower.load_state_dict(checkpoint['query_tower'])
doc_tower.load_state_dict(checkpoint['doc_tower'])
print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['val_loss']:.4f}")

# Load validation data
print("Loading validation data...")
val_df = pd.read_parquet('ms_marco_validation.parquet')
val_dataset = MSMARCOTripletDataset(
    df=val_df,
    max_query_len=config.MAX_QUERY_LEN,
    max_doc_len=config.MAX_DOC_LEN
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.TWOTOWERS_BATCH_SIZE,
    shuffle=False
)

# Calculate accuracy
print("\nCalculating validation accuracy...")
correct = 0
total = 0

qry_tower.eval()
doc_tower.eval()

with torch.no_grad():
    for batch in tqdm(val_dataloader):
        query_ids = batch['query_ids'].to(dev)
        pos_doc_ids = batch['pos_doc_ids'].to(dev)
        neg_doc_ids = batch['neg_doc_ids'].to(dev)
        
        # Get embeddings
        query_emb = qry_tower(query_ids)
        pos_doc_emb = doc_tower(pos_doc_ids)
        neg_doc_emb = doc_tower(neg_doc_ids)
        
        # Calculate similarities
        pos_sim = model.cosine_similarity(query_emb, pos_doc_emb)
        neg_sim = model.cosine_similarity(query_emb, neg_doc_emb)
        
        # Count where positive similarity > negative similarity
        correct += torch.sum(pos_sim > neg_sim).item()
        total += len(query_ids)

accuracy = (correct / total) * 100
print(f"\nValidation Accuracy: {accuracy:.2f}%")
print(f"Correct predictions: {correct}")
print(f"Total predictions: {total}") 