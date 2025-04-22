from engine.text import setup_language_models
from engine.data import MSMarcoDataset
from engine.model import Encoder, TripletLoss
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def run_sanity_check(q_vectors, pos_vectors, neg_vectors):
    B, K, D_out = neg_vectors.shape

    with torch.no_grad():
        q_pos_sim = torch.cosine_similarity(q_vectors, pos_vectors)
        q_neg_sim = torch.cosine_similarity(q_vectors, neg_vectors.view(B*K, D_out))

    logger.info(f'q_pos_sim: {q_pos_sim}')
    logger.info(f'q_neg_sim: {q_neg_sim}')


if __name__ == '__main__':
    tokeniser, w2v_model = setup_language_models()

    print(tokeniser.tokenise_string('hello world... 333'))

    train_ds = MSMarcoDataset('train')
    query, pos_docs, neg_docs = train_ds[0]
    print(query)
    print(pos_docs)
    print(neg_docs)

    # Tokenise, to be done in the dataloader
    query_token = tokeniser.tokenise_string(query)
    assert len(pos_docs) == 1, 'Only one positive document.'
    pos_token = tokeniser.tokenise_string(pos_docs[0])
    assert len(neg_docs) == 10, '10 negative documents.'
    neg_tokens = [tokeniser.tokenise_string(i) for i in neg_docs]

    print(query_token)
    print(pos_token)
    print(neg_tokens)
    # This is the backwards mapping for debugging
    print(tokeniser.tokens_to_words(pos_token))

    # Get pooled word embeddings (bag of words) for queries and all documents,
    # want shapes (1, D)
    q_embedding = w2v_model.forward(torch.tensor(query_token)).mean(dim=0)
    pos_embedding = w2v_model.forward(torch.tensor(pos_token)).mean(dim=0)
    # Stack negative embeddings (K, D)
    neg_embeddings = torch.stack(
        [w2v_model.forward(torch.tensor(i)).mean(dim=0) for i in neg_tokens])

    # The dataloader will handle the batching, so want to get the shape (B, 1, D) or (B, K, D)
    batch_q_embedding = q_embedding.unsqueeze(0)
    batch_pos_embedding = pos_embedding.unsqueeze(0)
    batch_neg_embeddings = neg_embeddings
    B = 1
    K = 10
    # Encode all queries
    D_in = w2v_model.embedding_dim
    D_hidden = 100
    D_out = 100
    query_encoder = Encoder(
        input_dim=D_in, hidden_dim=D_hidden, output_dim=D_out)

    # All these have shapes (B, D_out)
    q_vectors = query_encoder.forward(batch_q_embedding)
    pos_vectors = query_encoder.forward(batch_pos_embedding)
    # Batch compute this, neg_vectors which has shape (B, K, D_out)
    neg_vectors = query_encoder.forward(batch_neg_embeddings.view(B*K, D_in))

    # Permute to (B, D_out, K) for the triplet loss
    neg_vectors = neg_vectors.view(B, K, D_out)

    # Then do the triplet loss
    triplet_loss = TripletLoss()
    loss = triplet_loss(q_vectors, pos_vectors, neg_vectors.permute(0, 2, 1))
    print(loss)


