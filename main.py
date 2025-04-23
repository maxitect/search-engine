import logging
import os
from datetime import datetime
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from engine.data import MSMarcoDataset
from engine.model import Encoder, TripletLoss
from engine.text import setup_language_models, MeanPooledWordEmbedder
from engine.utils import get_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        num_correct = (
            torch.cat([q_pos_sim[:, None], q_neg_sim], 1).argmax(axis=1) == 0
        ).sum()

    # logger.info(f'q_pos_sim: {q_pos_sim}')
    # logger.info(f'q_neg_sim: {q_neg_sim}')

    # Calculate averaged accuracy over the batch
    accuracy = num_correct/B
    return accuracy


class Trainer:
    def __init__(
        self, batch_size: int, num_negative_samples: int,
        device: torch.device,
        log_to_wandb: bool = False,
    ):

        self.batch_size = batch_size
        self.K = num_negative_samples
        self.device = device
        self.log_to_wandb = log_to_wandb

        # Set up datasets and dataloaders
        self.train_ds = MSMarcoDataset('train', num_negative_samples)
        self.val_ds = MSMarcoDataset('validation', num_negative_samples)

        self.train_dl = DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        self.val_dl = DataLoader(
            self.val_ds, batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

        # Set up Mean Pooled sentence embedding model
        self.tokeniser, self.w2v_model = setup_language_models()

        self.sentence_embedder = MeanPooledWordEmbedder(
            self.tokeniser,
            self.w2v_model,
            device,
        )
        self.embed_fn = self.sentence_embedder.embed_string

        self.collate_fn = partial(
            self.collate_fn_marco,
            embed_fn=self.embed_fn,
        )

    def collate_fn_marco(self, batch, embed_fn) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Collates samples from MSMarcoDataset into batches of mean-pooled embeddings.

        Args:
            batch: A list of tuples of length batch_size, where each tuple is
                (query: str, pos_docs: list[str], neg_docs: list[str]).
                Assumes len(pos_docs) == 1.
            embed_fn: The embedding function.

        Returns:
            A tuple of tensors:
            - batch_q_embedding: (B, D_in)
            - batch_pos_embedding: (B, D_in)
            - batch_neg_embeddings: (B, K, D_in)
        """
        batch_size = len(batch)

        # Unzip the batch
        queries, pos_docs, neg_docs = zip(*batch)

        batch_q_embedding = torch.stack([
            embed_fn(q)
            for q in queries
        ])

        batch_pos_embedding = torch.stack([
            embed_fn(p)
            for p in pos_docs
        ])

        batch_neg_embeddings = []
        for b_idx in range(batch_size):
            batch_neg_embeddings.append(
                torch.stack([embed_fn(n) for n in neg_docs[b_idx]]),
            )
        batch_neg_embeddings = torch.stack(batch_neg_embeddings)

        return batch_q_embedding, batch_pos_embedding, batch_neg_embeddings

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
            batch_q_embedding, batch_pos_embedding, batch_neg_embeddings = batch
            batch_q_embedding = batch_q_embedding.to(self.device)
            batch_pos_embedding = batch_pos_embedding.to(self.device)
            batch_neg_embeddings = batch_neg_embeddings.to(self.device)

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
            loss.backward()
            optimiser.step()

            # Gather data and report
            running_loss += loss.item()
            if batch_idx % batches_print_frequency == (batches_print_frequency - 1):
                accuracy = run_sanity_check(
                    q_vectors, pos_vectors, neg_vectors,
                )

                # loss per batch
                last_loss = running_loss / batches_print_frequency
                logger.info(
                    f'  batch {batch_idx + 1} loss: {last_loss} accuracy: {accuracy}',
                )
                running_loss = 0.

        return last_loss, accuracy

    def validate(
        self,
        query_encoder: Encoder,
        doc_encoder: Encoder,
        loss_fn: TripletLoss,
    ):

        losses = []
        accuracies = []

        query_encoder = query_encoder.to(self.device)
        doc_encoder = doc_encoder.to(self.device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dl)):
                # Zero your gradients for every batch!
                batch_q_embedding, batch_pos_embedding, batch_neg_embeddings = batch
                batch_q_embedding = batch_q_embedding.to(self.device)
                batch_pos_embedding = batch_pos_embedding.to(self.device)
                batch_neg_embeddings = batch_neg_embeddings.to(self.device)

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

                accuracy = run_sanity_check(
                    q_vectors, pos_vectors, neg_vectors,
                )
                accuracies.append(accuracy)
                losses.append(loss.item())

        return sum(losses)/len(losses), sum(accuracies)/len(accuracies)

    def train(
        self,
        epochs: int,
        query_encoder: Encoder,
        doc_encoder: Encoder,
        triplet_loss: TripletLoss,
        optimiser: torch.optim.Optimizer,
        batches_print_frequency: int = 200,
    ):
        if self.log_to_wandb:
            run = wandb.init(
                entity='kwokkenton-individual',
                project='mlx-week2-search-engine',
                config={
                    'embeddings': 'self-trained',
                    'model': 'towers-mlp',
                    'epochs': epochs,
                    'batch_size': self.batch_size,
                    'num_negative_samples': self.K,
                    'lr': lr,
                    'D_hidden': D_hidden,
                    'D_out': D_out,
                },
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
                artifact = wandb.Artifact('towers_mlp', type='checkpoint')
                artifact.add_file(checkpoint_path)
                wandb.run.log_artifact(artifact)


if __name__ == '__main__':
    # Training configs
    batch_size = 32
    K = 20
    lr = 5e-4

    # Model configs
    D_hidden = 100
    D_out = 100

    log_to_wandb = True
    device = get_device()
    trainer = Trainer(
        batch_size,
        num_negative_samples=K,
        log_to_wandb=log_to_wandb,
        device=device,
    )

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
    trainer.train(
        epochs=10,
        query_encoder=query_encoder,
        doc_encoder=doc_encoder,
        triplet_loss=triplet_loss,
        optimiser=optimiser,
        batches_print_frequency=200,
    )
