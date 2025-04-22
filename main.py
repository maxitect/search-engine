from engine.text import setup_language_models
from engine.data import MSMarcoDataset
from engine.model import Encoder, TripletLoss
import torch
import logging
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_sanity_check(
    query, pos_docs, neg_docs,
    query_token, pos_token, neg_tokens,
    q_vectors, pos_vectors, neg_vectors,
):

    B, K, D_out = neg_vectors.shape

    with torch.no_grad():
        q_pos_sim = torch.cosine_similarity(q_vectors, pos_vectors)
        q_neg_sim = torch.cosine_similarity(
            q_vectors, neg_vectors.view(B*K, D_out),
        )

    logger.info(f'q_pos_sim: {q_pos_sim}')
    logger.info(f'q_neg_sim: {q_neg_sim}')

# def collate_fn(data: list[tuple[str, list[str], list[str]]]):
#     # Get batched sequences from dataloader, outputs are the embeddings already
#     query_docs, pos_docs, neg_docs = zip(*data)
#     # Then apply
#     pass


def embed_string(q, tokeniser, w2v_model):
    with torch.no_grad():
        token = tokeniser.tokenise_string(q)
        embedding = w2v_model.forward(torch.tensor(token)).mean(dim=0)
    return embedding


def embed_list(docs, tokeniser, w2v_model):
    return torch.stack([embed_string(doc, tokeniser, w2v_model) for doc in docs])


class Trainer:
    def __init__(self, batch_size: int, num_negative_samples: int):
        self.tokeniser, self.w2v_model = setup_language_models()
        self.train_ds = MSMarcoDataset('train', num_negative_samples)
        self.val_ds = MSMarcoDataset('validation', num_negative_samples)
        self.batch_size = batch_size

        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_dl = DataLoader(
            self.val_ds, batch_size=self.batch_size,
            shuffle=False,
        )

        self.K = num_negative_samples

    def train_one_epoch(self, query_encoder, doc_encoder, loss_fn, optimiser):

        for i, batch in enumerate(self.train_dl):
            # Get batched sequences from dataloader
            # Then apply
            # Tokenise, to be done in the dataloader
            # Encode all queries
            queries, pos_docs, neg_docs = batch
            batch_q_embedding = embed_list(
                queries, self.tokeniser, self.w2v_model,
            )
            # assert len(pos_docs) == 1, 'Only one positive document.'
            batch_pos_embedding = embed_list(
                pos_docs, self.tokeniser, self.w2v_model,
            )
            # assert len(neg_docs) == 10, '10 negative documents.'

            batch_neg_embeddings = []
            for b_idx in range(self.batch_size):
                batch_neg_embeddings.append(
                    embed_list(
                        neg_docs[b_idx], self.tokeniser, self.w2v_model,
                    ),
                )
            batch_neg_embeddings = torch.stack(batch_neg_embeddings)

            # print(query_token)
            # print(pos_token)
            # print(neg_tokens)
            # This is the backwards mapping for debugging
            # print(self.tokeniser.tokens_to_words(pos_token))

            # Get pooled word embeddings (bag of words) for queries and all documents,
            # want shapes (1, D)
            # q_embedding = self.w2v_model.forward(torch.tensor(query_token)).mean(dim=0)
            # pos_embedding = self.w2v_model.forward(torch.tensor(pos_token)).mean(dim=0)
            # # Stack negative embeddings (K, D)
            # neg_embeddings = torch.stack(
            #     [self.w2v_model.forward(torch.tensor(i)).mean(dim=0) for i in neg_tokens])

            # The dataloader will handle the batching, so want to get the shape (B, 1, D) or (B, K, D)
            # batch_q_embedding = q_embedding.unsqueeze(0)
            # batch_pos_embedding = pos_embedding.unsqueeze(0)
            # batch_neg_embeddings = neg_embeddings

            # All these have shapes (B, D_out)
            q_vectors = query_encoder.forward(batch_q_embedding)
            pos_vectors = doc_encoder.forward(batch_pos_embedding)
            # Batch compute this, neg_vectors which has shape (B, K, D_out)
            neg_vectors = doc_encoder.forward(
                batch_neg_embeddings.view(self.batch_size*self.K, -1),
            )

            # Permute to (B, D_out, K) for the triplet loss
            neg_vectors = neg_vectors.view(self.batch_size, self.K, -1)

            # Then do the triplet loss
            loss = loss_fn(
                q_vectors, pos_vectors,
                neg_vectors.permute(0, 2, 1),
            )
            print(loss)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            checkpoint = {}
            break

        return checkpoint

    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch()


if __name__ == '__main__':

    # Training configs
    batch_size = 5
    K = 10
    lr = 0.001

    # Model configs
    D_hidden = 100
    D_out = 100

    trainer = Trainer(batch_size, num_negative_samples=K)

    D_in = trainer.w2v_model.embedding_dim

    query_encoder = Encoder(
        input_dim=D_in, hidden_dim=D_hidden, output_dim=D_out,
    )
    doc_encoder = Encoder(
        input_dim=D_in, hidden_dim=D_hidden, output_dim=D_out,
    )

    optimiser = torch.optim.Adam(
        list(query_encoder.parameters()) + list(doc_encoder.parameters()), lr=lr,
    )

    triplet_loss = TripletLoss()

    trainer.train_one_epoch(query_encoder, doc_encoder,
                            triplet_loss, optimiser)

    # print(tokeniser.tokenise_string('hello world... 333'))

    # train_ds = MSMarcoDataset('train')
    # query, pos_docs, neg_docs = train_ds[0]
    # print(query)
    # print(pos_docs)
    # print(neg_docs)
