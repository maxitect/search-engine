import random
import pandas as pd
from src.dataset import MSMARCOTripletDataset
from src.models.skipgram import negative_sampling_loss
from src.models.twotowers import triplet_loss_function
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
def topk(model):
    words = [
        'anarchism',
        'monarchy',
        'democracy',
        'computer',
        'laptop',
        'beach',
        'mountain',
        'river',
        'ocean',
        'algorithm',
        'mathematics',
        'physics',
        'chemistry',
        'football',
        'rugby',
        'violin',
        'guitar',
        'hospital',
        'medicine',
        'university'
    ]
    word = words[random.randint(0, len(words) - 1)]
    idx = vocab_to_int[word]
    vec = model.in_embed.weight[idx].detach()
    with torch.no_grad():
        vec = torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1)
        emb = torch.nn.functional.normalize(
            model.in_embed.weight.detach(),
            p=2, dim=1
        )
        sim = torch.matmul(emb, vec.squeeze())
        top_val, top_idx = torch.topk(sim, 6)
        print(f'\nTop 5 words similar to {word}:')
        count = 0
        for i, idx in enumerate(top_idx):
            word = int_to_vocab[idx.item()]
            sim = top_val[i].item()
            print(f'  {word}: {sim:.4f}')
            count += 1
            if count == 5:
                break


def evaluate_model(model, word_freq, data_loader, name="val"):
    model.eval()
    total_loss = 0
    batch_count = 0
    num_neg_samples = config.NEGATIVE_SAMPLES

    with torch.no_grad():
        for ipt, trg in data_loader:
            ipt, trg = ipt.to(dev), trg.to(dev)
            batch_size = ipt.size(0)

            # Sample negative words
            neg_samples = torch.multinomial(
                torch.tensor(word_freq, device=dev),
                batch_size * num_neg_samples,
                replacement=True
            ).view(batch_size, num_neg_samples)

            # Get embeddings and calculate loss
            input_embeds, pos_embeds, neg_embeds = model(
                ipt, trg, neg_samples)
            loss = negative_sampling_loss(
                input_embeds, pos_embeds, neg_embeds)

            total_loss += loss.item()
            batch_count += 1

    avg_loss = total_loss / batch_count
    print(f"{name.capitalize()} Loss: {avg_loss:.4f}")
    return avg_loss


# Evaluate two tower model with test queries
def evaluate_progress(qry_tower, doc_tower, step):
    qry_tower.eval()
    doc_tower.eval()

    print(f"\n--- Evaluation at {step} ---")

    # Create a dataset for evaluation using our existing MSMARCOTripletDataset
    eval_dataset = MSMARCOTripletDataset(
        df=test_df,
        max_query_len=config.MAX_QUERY_LEN,
        max_doc_len=config.MAX_DOC_LEN
    )

    if len(eval_dataset) == 0:
        print("No evaluation examples available")
        return

    # Get a random query from the test set
    random_idx = random.randint(0, len(test_df) - 1)
    query = test_df.iloc[random_idx]['queries']
    documents = test_df.iloc[random_idx]['documents']
    labels = test_df.iloc[random_idx]['labels']

    print(f"\nQuery: {query}")

    # Separate positive and negative passages
    pos_indices = [i for i, label in enumerate(labels) if label == 1]
    neg_indices = [i for i, label in enumerate(
        labels) if label == 0][:5]  # Limit to 5 negatives

    if not pos_indices:
        print("  No positive passages found for this query.")
        return

    print(
        f"  Found {len(pos_indices)} positive and "
        f"{len(neg_indices)} negative passages"
    )

    # Process query
    query_ids = eval_dataset._tokenise(
        query, eval_dataset.max_query_len).unsqueeze(0).to(dev)
    with torch.no_grad():
        query_embedding = qry_tower(query_ids)

    # Process all passages
    similarities = []

    # Process positive passages
    for i in pos_indices:
        doc = documents[i]
        doc_ids = eval_dataset._tokenise(
            doc, eval_dataset.max_doc_len).unsqueeze(0).to(dev)

        with torch.no_grad():
            doc_embedding = doc_tower(doc_ids)
            similarity = torch.sum(
                query_embedding * doc_embedding, dim=1).item()
            similarities.append((doc, similarity, "POSITIVE"))

    # Process negative passages
    for i in neg_indices:
        doc = documents[i]
        doc_ids = eval_dataset._tokenise(
            doc, eval_dataset.max_doc_len).unsqueeze(0).to(dev)

        with torch.no_grad():
            doc_embedding = doc_tower(doc_ids)
            similarity = torch.sum(
                query_embedding * doc_embedding, dim=1).item()
            similarities.append((doc, similarity, "NEGATIVE"))

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
        f"out of {len(pos_indices)}")

    # Calculate test loss
    test_loss = 0.0
    num_triplets = 0

    for pos_idx in pos_indices:
        for neg_idx in neg_indices:
            pos_doc = documents[pos_idx]
            neg_doc = documents[neg_idx]

            query_ids = eval_dataset._tokenise(
                query, eval_dataset.max_query_len).unsqueeze(0).to(dev)
            pos_doc_ids = eval_dataset._tokenise(
                pos_doc, eval_dataset.max_doc_len).unsqueeze(0).to(dev)
            neg_doc_ids = eval_dataset._tokenise(
                neg_doc, eval_dataset.max_doc_len).unsqueeze(0).to(dev)

            with torch.no_grad():
                query_emb = qry_tower(query_ids)
                pos_doc_emb = doc_tower(pos_doc_ids)
                neg_doc_emb = doc_tower(neg_doc_ids)

                loss = triplet_loss_function(
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

    return avg_test_loss
