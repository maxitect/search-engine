import random
import pandas as pd
from src.dataset import MSMARCODataset
from src.model import triplet_loss
import src.config as config
import torch
import pickle

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


vocab_to_int = pickle.load(
    open(config.VOCAB_TO_ID_PATH, 'rb'))
int_to_vocab = pickle.load(
    open(config.ID_TO_VOCAB_PATH, 'rb'))


# Load test data from parquet file
test_df = pd.read_parquet('ms_marco_test.parquet')


# Evaluate word similarity for skipgram training
def topk(mFoo):

    idx = vocab_to_int['computer']
    vec = mFoo.emb.weight[idx].detach()
    with torch.no_grad():

        vec = torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1)
        emb = torch.nn.functional.normalize(
            mFoo.emb.weight.detach(), p=2, dim=1)
        sim = torch.matmul(emb, vec.squeeze())
        top_val, top_idx = torch.topk(sim, 6)
        print('\nTop 5 words similar to "computer":')
        count = 0
        for i, idx in enumerate(top_idx):
            word = int_to_vocab[idx.item()]
            sim = top_val[i].item()
            print(f'  {word}: {sim:.4f}')
            count += 1
            if count == 5:
                break


# Evaluate two tower model with test queries
def evaluate_progress(qry_tower, doc_tower, step):
    qry_tower.eval()
    doc_tower.eval()

    print(f"\n--- Evaluation at step {step} ---")

    # Create a dataset for evaluation
    eval_dataset = MSMARCODataset(
        queries=test_df['queries'].tolist(),
        documents=test_df['documents'].tolist(),
        labels=test_df['labels'].tolist(),
        vocab_to_int=vocab_to_int,
        max_query_len=config.MAX_QUERY_LEN,
        max_doc_len=config.MAX_DOC_LEN
    )

    index = random.randint(0, len(test_df['queries']) - 1)
    query = test_df['queries'][index]
    documents = test_df['documents'][index]
    labels = test_df['labels'][index]

    print(f"\nQuery: {query}")

    # Get passages for this query
    query_data = test_df[test_df['queries'] == query]

    # Separate positive and negative passages
    pos_query = query_data[labels == 1]
    positive_indices = pos_query.index.tolist()
    neg_query = query_data[labels == 0]
    negative_indices = neg_query.index.tolist()[:5]  # Limit to 5 negatives

    if not positive_indices:
        print("  No positive passages found for this query.")

    print(
        f"  Found {len(positive_indices)} positive and "
        f"{len(negative_indices)} negative passages"
    )

    # Get query embedding once
    sample = eval_dataset[positive_indices[0]]
    query_ids = sample['query_ids'].unsqueeze(0).to(dev)
    with torch.no_grad():
        query_embedding = qry_tower(query_ids)

    # Process all passages
    similarities = []

    # Process positive passages
    for idx in positive_indices:
        sample = eval_dataset[idx]
        doc_ids = sample['doc_ids'].unsqueeze(0).to(dev)

        with torch.no_grad():
            doc_embedding = doc_tower(doc_ids)
            similarity = torch.sum(
                query_embedding * doc_embedding, dim=1).item()
            similarities.append(
                (documents[idx], similarity, "POSITIVE"))

    # Process negative passages
    for idx in negative_indices:
        sample = eval_dataset[idx]
        doc_ids = sample['doc_ids'].unsqueeze(0).to(dev)

        with torch.no_grad():
            doc_embedding = doc_tower(doc_ids)
            similarity = torch.sum(
                query_embedding * doc_embedding, dim=1).item()
            similarities.append(
                (documents[idx], similarity, "NEGATIVE"))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Print top results
    for i, (passage, sim, label) in enumerate(similarities[:5]):
        print(f"  {i+1}. [{sim:.4f}] [{label}] {passage[:100]}...")

    # Calculate ranking metrics
    best_pos_rank = next(
        (i+1 for i, (_, _, label) in enumerate(similarities)
            if label == "POSITIVE"),
        len(similarities)+1
    )
    print(f"  Best positive passage rank: {best_pos_rank}")

    # Count how many positive passages are in top-5
    top5_positives = sum(
        1 for _, _, label in similarities[:5] if label == "POSITIVE")
    print(
        f"  Positive passages in top-5: {top5_positives} "
        f"out of {len(positive_indices)}")

    # Calculate test loss
    test_loss = 0.0
    num_triplets = 0

    for pos_idx in positive_indices:
        for neg_idx in negative_indices:
            pos_sample = eval_dataset[pos_idx]
            neg_sample = eval_dataset[neg_idx]

            query_ids = pos_sample['query_ids'].unsqueeze(0).to(dev)
            pos_doc_ids = pos_sample['doc_ids'].unsqueeze(0).to(dev)
            neg_doc_ids = neg_sample['doc_ids'].unsqueeze(0).to(dev)

            with torch.no_grad():
                query_emb = qry_tower(query_ids)
                pos_doc_emb = doc_tower(pos_doc_ids)
                neg_doc_emb = doc_tower(neg_doc_ids)

                loss = triplet_loss(
                    query_emb,
                    pos_doc_emb,
                    neg_doc_emb,
                    margin=config.MARGIN
                )
                test_loss += loss.item()
                num_triplets += 1

    avg_test_loss = test_loss / num_triplets if num_triplets > 0 else 0
    print(f"  Test loss for this query: {avg_test_loss:.4f}")

    qry_tower.train()
    doc_tower.train()
