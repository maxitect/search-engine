import logging
import random

from datasets import load_dataset as hf_load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ms_marco() -> DatasetDict:
    ds = hf_load_dataset('microsoft/ms_marco', 'v1.1')
    return ds


class MSMarcoDataset(Dataset):
    def __init__(self, split: str, mode: str, num_negative_samples: int = 10):
        """Dataset for training a retrieval model, giving a query and a list of
            positive and negative answers. Returns a single positive samples and 
            a number of negative samples in raw format (strings or list of 
            strings).

            This is for maximising the flexibility of downstream processing, 
            which can be done in the dataloader.

        Args:
            split (str): The split to load, must be one of "train",
            "validation", or "test".
            mode (str): The mode to load, must be one of "hard" or "random".
            num_negative_samples (int): The number of negative samples to load.

        Raises:
            ValueError: If the split is not valid.
        """
        super().__init__()

        if split not in ['train', 'validation', 'test']:
            raise ValueError(f'Invalid split: {split}')
        
        if mode not in ['hard', 'random']:
            raise ValueError(f'Invalid mode: {mode}')
        
        ds = load_ms_marco()[split]
        logger.info(f'Loaded {split} dataset')

        # Get rows with 1 positive answer
        self.rows_with_k_answers = self.rows_k_answers(ds, k=1)
        self.ds = ds.select(self.rows_with_k_answers)
        logger.info(f'Selected {len(self.ds)} rows with 1 answer')
        self.num_negative_samples = num_negative_samples
        self.mode = mode
    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return str(self.ds)

    def __getitem__(self, idx):
        """ Get training instance

        Returns a tuple of (query: str,
            positive_answer: str,
            negative_answers: List[str])
        """
        # Want positive and negative samples
        if self.mode == 'hard':
            query, positive_answer, negative_answers = self._get_entry_item_hard(idx)

        elif self.mode == 'random':
            query, positive_answer, negative_answers = self._get_entry_item_random(idx)

        return query, positive_answer, negative_answers
    
    def _get_entry_item_random(self, idx):
        # Treat all 'passage_texts' as positive answers
        query = self.ds[idx]['query']
        positives = self.ds[idx]['passages']['passage_text']

        # Randomly select one positive answer
        positive_answer = positives[random.randint(0, len(positives) - 1)]

        # Get random negatives
        negatives = self._get_random_negative_samples(
            idx,
            num_samples=self.num_negative_samples,
        )
        return query, positive_answer, negatives

    def _get_entry_item_hard(self, idx):
        row = self.ds[idx]
        query = row['query']

        # Get positive and negative answers
        positive_answer = []
        negative_answers = []

        for text, is_selected in zip(
            row['passages']['passage_text'],
            row['passages']['is_selected'],
        ):
            if is_selected:
                positive_answer.append(text)
            else:
                negative_answers.append(text)
        assert len(positive_answer) == 1, 'Only one positive answer'

        # Fill to num_negative_samples
        num_hard_negatives = len(negative_answers)
        if num_hard_negatives >= self.num_negative_samples:
            # Too many hard negatives: randomly select from the negative answers
            negative_answers = random.sample(
                negative_answers, self.num_negative_samples,
            )
        else:
            # Not enough hard negatives: get random negatives
            num_random_negatives = self.num_negative_samples - num_hard_negatives
            random_answers = self._get_random_negative_samples(
                idx,
                num_samples=num_random_negatives,
            )
            negative_answers.extend(random_answers)

        return query, positive_answer[0], negative_answers

    def rows_k_answers(self, ds, k=1):
        # Returns sorted list of rows with k answers
        num_selected = []
        for idx, row in enumerate(ds):
            num_selected.append(sum(row['passages']['is_selected']))

        row_idxs = [i for i, x in enumerate(num_selected) if x == k]
        return row_idxs

    def _get_random_negative_samples(
        self,
        positive_idx: int,
        num_samples: int,
    ) -> list[str]:
        # Get random negative samples from the dataset
        negative_samples = []

        for _ in range(num_samples):
            random_ds_row = self.ds[random.randint(0, len(self.ds) - 1)]
            # This is to avoid collisions
            while random_ds_row == positive_idx:
                random_ds_row = self.ds[random.randint(0, len(self.ds) - 1)]

            passage_text = random_ds_row['passages']['passage_text']
            negative_samples.append(
                passage_text[random.randint(0, len(passage_text) - 1)],
            )
        return negative_samples

def collate_fn_marco(batch, embed_fn, padding=False) -> tuple[
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
        embed_fn: The mean-pooled embedding function.
        padding (bool): Whether to pad the embeddings sequences to the same length.

    Returns:
        A tuple of tensors:
        If padding is False:
            - batch_q_embedding: (B, D_in)
            - batch_pos_embedding: (B, D_in)
            - batch_neg_embeddings: (B, K, D_in)

        If padding is True:
            - batch_q_embedding: (B, max_len, D_in)
            - batch_pos_embedding: (B, max_len, D_in)
            - batch_neg_embeddings: (B, max_len, K, D_in)
    """
    batch_size = len(batch)

    # Unzip the batch
    queries, pos_docs, neg_docs = zip(*batch)

    q_embeddings = [embed_fn(q) for q in queries]
    p_embeddings = [embed_fn(p) for p in pos_docs]

    if not padding:
        batch_q_embedding = torch.stack(q_embeddings)
        batch_pos_embedding = torch.stack(p_embeddings)

        batch_neg_embeddings = []
        for b_idx in range(batch_size):
            batch_neg_embeddings.append(
                torch.stack([embed_fn(n) for n in neg_docs[b_idx]]),
            )
        batch_neg_embeddings = torch.stack(batch_neg_embeddings)
    else:
        # Pad the embeddings with zeros
        batch_q_embedding = pad_sequence(q_embeddings, batch_first=True)
        batch_pos_embedding = pad_sequence(p_embeddings, batch_first=True)

        for b_idx in range(batch_size):
            neg_embeddings = pad_sequence(
                [embed_fn(n) for n in neg_docs[b_idx]],
                batch_first=True,
            )

            batch_neg_embeddings.append(neg_embeddings)

        batch_neg_embeddings = pad_sequence(batch_neg_embeddings, batch_first=True)

    return batch_q_embedding, batch_pos_embedding, batch_neg_embeddings

