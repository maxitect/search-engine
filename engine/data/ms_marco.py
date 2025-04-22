import logging
import random

from datasets import load_dataset as hf_load_dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ms_marco() -> DatasetDict:
    ds = hf_load_dataset('microsoft/ms_marco', 'v1.1')
    return ds


class MSMarcoDataset(Dataset):
    def __init__(self, split: str, num_negative_samples: int = 10):
        """Dataset for training a retrieval model, giving a query and a list of
            positive and negative answers. All hard negatives are returned, and
            a number of other negatives are sampled randomly from the dataset.

        Args:
            split (str): The split to load, must be one of "train",
            "validation", or "test".

        Raises:
            ValueError: If the split is not valid.
        """
        super().__init__()

        if split not in ['train', 'validation', 'test']:
            raise ValueError(f'Invalid split: {split}')
        ds = load_ms_marco()[split]
        logger.info(f'Loaded {split} dataset')
        self.rows_with_k_answers = self.rows_k_answers(ds, k=1)
        self.ds = ds.select(self.rows_with_k_answers)
        logger.info(f'Selected {len(self.ds)} rows with 1 answer')
        self.num_negative_samples = num_negative_samples

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
        query, positive_answer, negative_answers = self._get_entry_item(idx)
        num_hard_negatives = len(negative_answers)
        assert num_hard_negatives < self.num_negative_samples, (
            'Num negatives is less than num hard negatives'
        )

        num_random_negatives = self.num_negative_samples - num_hard_negatives
        random_answers = self._get_random_negative_samples(
            idx,
            num_samples=num_random_negatives,
        )
        negative_answers.extend(random_answers)
        return query, positive_answer, negative_answers

    def _get_entry_item(self, idx):

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
