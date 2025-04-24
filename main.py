import logging
import os
from datetime import datetime
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from engine.data import MSMarcoDataset, collate_fn_marco
from engine.model import Encoder, RNNEncoder, TripletLoss
from engine.text import setup_language_models, MeanPooledWordEmbedder, GensimWord2Vec
from engine.utils import get_device, get_wandb_checkpoint_path
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NUM_WORKERS = 4


def print_validation_results(
    query: str,
    similarities: torch.Tensor,
    docs_strings: list[str],
    positive_idx: int = 0,
):
    """Print validation results in a nice format.

    Args:
        query: The search query
        similarities: Tensor of similarity scores
        docs_strings: List of document strings
        positive_idx: Index of the positive document (default 0)
    """
    # Convert similarities to list of floats
    similarities = similarities.cpu().numpy().tolist()
    docs_ids = torch.arange(len(similarities))

    # Create a formatted string
    output = []
    output.append('=' * 80)
    output.append(f'Query: {query}')
    output.append('-' * 80)

    # Sort documents by similarity
    sorted_docs = sorted(
        zip(
            similarities, docs_strings,
            docs_ids,
        ), key=lambda x: x[0], reverse=True,
    )

    for i, (sim, doc, doc_id) in enumerate(sorted_docs):
        # Mark positive document
        is_positive = (doc_id == positive_idx)
        marker = '✅' if is_positive else '  '

        # Format the output
        output.append(f'{marker} Similarity: {sim:.4f}')
        output.append(f'   Document: {doc}')
        output.append('-' * 80)

    # Join and print
    # print("\n".join(output))
    logger.info('\n'.join(output))


def run_sanity_check(
    q_vectors, pos_vectors, neg_vectors,
):
    """
    Run a sanity check on the embeddings.

    Args:
        q_vectors: (B, D_out)
        pos_vectors: (B, D_out)
        neg_vectors: (B, K, D_out)
    """

    # This is the backwards mapping for debugging
    # print(self.tokeniser.tokens_to_words(pos_token))
    B, K, D_out = neg_vectors.shape
    accuracy = None
    neg_vectors = neg_vectors.permute(0, 2, 1)

    with torch.no_grad():
        q_pos_sim = torch.cosine_similarity(q_vectors, pos_vectors)
        q_neg_sim = torch.cosine_similarity(
            q_vectors[:, :, None], neg_vectors,
        )
        # Positive sample is at index 0, so check whether the argmax is 0
        # and subsequently count
        num_correct = (
            torch.cat([q_pos_sim[:, None], q_neg_sim], 1).argmax(axis=1) == 0
        ).sum()

    # Calculate averaged accuracy over the batch
    accuracy = num_correct / B
    return accuracy


class Trainer:
    def __init__(
        self, batch_size: int,
        num_negative_samples: int,
        mode: str,
        embeddings: str,
        device: torch.device,
        rnn: bool = False,
        log_to_wandb: bool = False,
    ):

        self.batch_size = batch_size
        self.K = num_negative_samples
        self.mode = mode
        self.embeddings = embeddings
        self.rnn = rnn
        self.device = device
        self.log_to_wandb = log_to_wandb

        # Set up datasets and dataloaders
        self.train_ds = MSMarcoDataset('train', mode, num_negative_samples)
        self.val_ds = MSMarcoDataset('validation', mode, num_negative_samples)

        if embeddings not in ['self-trained', 'word2vec-google-news-300']:
            raise ValueError(f'Invalid embeddings: {embeddings}')

        # Set up sentence embedding model
        # Either mean pooled or not pooled
        if embeddings == 'self-trained':
            if self.rnn:
                raise NotImplementedError(
                    'Padding is not implemented for self-trained embeddings',
                )
            self.tokeniser, self.w2v_model = setup_language_models()

            self.sentence_embedder = MeanPooledWordEmbedder(
                self.tokeniser,
                self.w2v_model,
                device,
            )
            self.embed_fn = self.sentence_embedder.embed_string
        elif embeddings == 'word2vec-google-news-300':
            self.gensim_w2v = GensimWord2Vec()

            if not self.rnn:
                self.embed_fn = self.gensim_w2v.get_mean_embedding
            else:
                self.embed_fn = self.gensim_w2v.get_sentence_embeddings

        # Change collate function based on whether padding is enabled
        self.collate_fn = partial(
            collate_fn_marco,
            embed_fn=self.embed_fn,
            padding=self.rnn,
        )

        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=NUM_WORKERS,
        )
        self.val_dl = DataLoader(
            self.val_ds, batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
            num_workers=NUM_WORKERS,
        )

    def train_one_epoch(
        self,
        query_encoder: Encoder,
        doc_encoder: Encoder,
        loss_fn: TripletLoss,
        optimiser: torch.optim.Optimizer,
        batches_print_frequency: int = 1000,
    ):

        running_loss = 0.
        last_loss = 0.

        query_encoder = query_encoder.to(self.device)
        doc_encoder = doc_encoder.to(self.device)

        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            # Zero your gradients for every batch!
            optimiser.zero_grad()
            q_vectors, pos_vectors, neg_vectors = self._vectors_from_batch(
                batch, query_encoder, doc_encoder,
            )
            # Then do the triplet loss
            # Permute to (B, D_out, K) for the triplet loss
            loss = loss_fn(
                q_vectors, pos_vectors,
                neg_vectors.permute(0, 2, 1),
            )
            loss.backward()
            optimiser.step()

            # Gather data and report
            running_loss += loss.item()
            if batch_idx % batches_print_frequency == (batches_print_frequency - 1):
                accuracy = run_sanity_check(
                    q_vectors, pos_vectors, neg_vectors,
                )

                self.validate_single(query_encoder, doc_encoder)

                # loss per batch
                last_loss = running_loss / batches_print_frequency
                logger.info(
                    f'  For batch {batch_idx + 1}, the loss is {last_loss} and the accuracy is {accuracy}',
                )
                running_loss = 0.

        return last_loss, accuracy

    def _vectors_from_batch(
        self,
        batch,
        query_encoder: Encoder,
        doc_encoder: Encoder,
    ):
        # Allow for variable batch sizes
        batch_size = len(batch[0])

        batch_q_embedding, batch_pos_embedding, batch_neg_embeddings = batch
        batch_q_embedding = batch_q_embedding.to(self.device)
        batch_pos_embedding = batch_pos_embedding.to(self.device)
        batch_neg_embeddings = batch_neg_embeddings.to(self.device)

        # All these have shapes (batch_size, D_out)
        q_vectors = query_encoder.forward(batch_q_embedding)
        pos_vectors = doc_encoder.forward(batch_pos_embedding)
        if not self.rnn:
            # Batch compute this, neg_vectors which has shape (batch_size, K, D_out)
            neg_vectors = doc_encoder.forward(
                batch_neg_embeddings.view(batch_size*self.K, -1),
            )
            neg_vectors = neg_vectors.view(batch_size, self.K, -1)
        else:
            # The RNN encoder handles the batching
            # Shape is (B, L, K, D_in), reshape to (B*K, L, D_in)
            _, seq_length, _, D_in = batch_neg_embeddings.shape
            batch_neg_embeddings = batch_neg_embeddings.permute(0, 2, 1, 3)
            neg_vectors = doc_encoder.forward(
                batch_neg_embeddings.reshape(
                    batch_size*self.K, seq_length, D_in),
            )
            neg_vectors = neg_vectors.view(batch_size, self.K, -1)
        return q_vectors, pos_vectors, neg_vectors

    def validate(
        self,
        query_encoder: Encoder,
        doc_encoder: Encoder,
        loss_fn: TripletLoss,
    ):
        # Run ablation here on validation set
        losses = []
        accuracies = []

        query_encoder = query_encoder.to(self.device)
        doc_encoder = doc_encoder.to(self.device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dl)):
                # Zero your gradients for every batch!
                q_vectors, pos_vectors, neg_vectors = self._vectors_from_batch(
                    batch, query_encoder, doc_encoder,
                )
                loss = loss_fn(
                    q_vectors, pos_vectors,
                    neg_vectors.permute(0, 2, 1),
                )
                accuracy = run_sanity_check(
                    q_vectors, pos_vectors, neg_vectors,
                )
                accuracies.append(accuracy)
                losses.append(loss.item())

        return sum(losses)/len(losses), sum(accuracies)/len(accuracies)

    def validate_single(
        self,
        query_encoder: Encoder,
        doc_encoder: Encoder,
    ):
        # Select random sample from the validation set
        dataset = self.val_dl.dataset
        random_idx = random.randint(0, len(dataset))
        query, pos_doc, neg_docs = dataset[random_idx]

        # Enforce positive doc is first
        docs_strings = [pos_doc] + neg_docs

        query_encoder = query_encoder.to(self.device)
        doc_encoder = doc_encoder.to(self.device)

        with torch.no_grad():
            # Collate function expects a list of tuples
            batch = self.collate_fn([(query, pos_doc, neg_docs)])
            q_vectors, pos_vectors, neg_vectors = self._vectors_from_batch(
                batch, query_encoder, doc_encoder,
            )
            # Compute cosine similarity
            docs_vectors = torch.cat([pos_vectors, neg_vectors.squeeze()])
            similarities = torch.cosine_similarity(
                q_vectors,
                docs_vectors,
                dim=-1,
            )

            print_validation_results(
                query, similarities, docs_strings, positive_idx=0,
            )

    def train(
        self,
        epochs: int,
        query_encoder: Encoder,
        doc_encoder: Encoder,
        triplet_loss: TripletLoss,
        optimiser: torch.optim.Optimizer,
        batches_print_frequency: int = 200,
    ):

        config = {
            'embeddings': self.embeddings,
            'mode': self.mode,
            'model': 'rnn' if self.rnn else 'towers-mlp',
            'epochs': epochs,
            'batch_size': self.batch_size,
            'num_negative_samples': self.K,
            'lr': lr,
            'D_hidden': D_hidden,
            'D_out': D_out,
        }

        logger.info(f'Training config: {config}')

        if self.log_to_wandb:
            run = wandb.init(
                entity='kwokkenton-individual',
                project='mlx-week2-search-engine',
                config=config,
            )

        for epoch in range(epochs):
            logger.info(f'Training: Epoch {epoch + 1} of {epochs}')
            train_loss, train_accuracy = self.train_one_epoch(
                query_encoder, doc_encoder,
                triplet_loss, optimiser, batches_print_frequency,
            )
            logger.info(
                f'Validating: Epoch {epoch + 1} of {epochs}.',
            )
            val_loss, val_accuracy = self.validate(
                query_encoder, doc_encoder,
                triplet_loss,
            )
            logger.info(
                f'Epoch {epoch + 1} of {epochs} train loss: {train_loss}'
                f'train accuracy: {train_accuracy} val loss: {val_loss}'
                f'val accuracy: {val_accuracy}',
            )
            if self.log_to_wandb:
                wandb.log({
                    'train/loss': train_loss,
                    'train/accuracy': train_accuracy,
                    'val/loss': val_loss,
                    'val/accuracy': val_accuracy,
                })

                checkpoint = {
                    'query_encoder_state_dict': query_encoder.state_dict(),
                    'doc_encoder_state_dict': doc_encoder.state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                }
                checkpoint_path = os.path.join(
                    wandb.run.dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                )
                torch.save(checkpoint, checkpoint_path)
                if self.rnn:
                    artifact = wandb.Artifact('towers_rnn', type='checkpoint')
                else:
                    artifact = wandb.Artifact('towers_mlp', type='checkpoint')
                artifact.add_file(checkpoint_path)
                wandb.run.log_artifact(artifact)


if __name__ == '__main__':
    # Training configs
    batch_size = 8
    K = 5
    lr = 1e-3
    mode = 'random'
    # embeddings = 'self-trained'
    embeddings = 'word2vec-google-news-300'
    num_epochs = 30
    # Model configs
    D_in = 300
    D_hidden = 200
    D_out = 100
    batches_print_frequency = 200
    log_to_wandb = True

    device = get_device()

    trainer = Trainer(
        batch_size,
        num_negative_samples=K,
        mode=mode,
        embeddings=embeddings,
        log_to_wandb=log_to_wandb,
        device=device,
        rnn=True,
    )

    # query_encoder = Encoder(
    #     input_dim=D_in, hidden_dim=D_hidden, output_dim=D_out,
    # )
    # doc_encoder = Encoder(
    #     input_dim=D_in, hidden_dim=D_hidden, output_dim=D_out,
    # )

    query_encoder = RNNEncoder(
        input_dim=D_in, hidden_dim=D_hidden, output_dim=D_out,
    )
    doc_encoder = RNNEncoder(
        input_dim=D_in, hidden_dim=D_hidden, output_dim=D_out,
    )

    if type(query_encoder) == RNNEncoder:
        assert trainer.rnn == True, 'RNN encoder requires padding.'

    optimiser = torch.optim.Adam(
        list(query_encoder.parameters()) + list(doc_encoder.parameters()), lr=lr,
    )

    # # Load previously trained model
    # checkpoint_path = get_wandb_checkpoint_path(
    #     'kwokkenton-individual/mlx-week2-search-engine/towers_mlp:v49',
    # )

    # # Load the model
    # checkpoint = torch.load(
    #     checkpoint_path, map_location=device, weights_only=True,
    # )
    # query_encoder.load_state_dict(checkpoint['query_encoder_state_dict'])
    # doc_encoder.load_state_dict(checkpoint['doc_encoder_state_dict'])

    triplet_loss = TripletLoss()
    trainer.train(
        epochs=num_epochs,
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        triplet_loss=triplet_loss,
        optimiser=optimiser,
        batches_print_frequency=batches_print_frequency,
    )
