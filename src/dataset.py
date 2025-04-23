import torch
from torch.utils.data import Dataset
import pickle
import collections
import numpy as np

import src.config as config
from src.utils.tokenise import preprocess


class Wiki(Dataset):
    def __init__(self, skip_gram=True):
        self.vocab_to_int = pickle.load(
            open(config.VOCAB_TO_ID_PATH, 'rb'))
        self.int_to_vocab = pickle.load(
            open(config.ID_TO_VOCAB_PATH, 'rb'))
        self.corpus = pickle.load(open(config.CORPUS_PATH, 'rb'))
        self.tokens = [self.vocab_to_int[word] for word in self.corpus]
        self.skip_gram = skip_gram

        # Calculate word frequency distribution for negative sampling
        self._calculate_word_frequencies()

    def _calculate_word_frequencies(self):
        """Calculate word frequencies for negative sampling"""
        word_counts = collections.Counter(self.corpus)
        vocab_size = len(self.vocab_to_int)

        # Initialize frequencies array with a small value to avoid zeros
        self.word_freqs = np.ones(vocab_size) * 1e-5

        # Fill in actual frequencies
        for word, count in word_counts.items():
            if word in self.vocab_to_int:
                self.word_freqs[self.vocab_to_int[word]] = count

        # Apply the 3/4 power as recommended in the paper
        self.word_freqs = np.power(self.word_freqs, 0.75)
        # Normalize to get probability distribution
        self.word_freqs = self.word_freqs / np.sum(self.word_freqs)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.skip_gram:
            center = self.tokens[idx]
            context = []
            for j in range(idx - 2, idx + 3):
                if j != idx and 0 <= j < len(self.tokens):
                    context.append((center, self.tokens[j]))
            if len(context) == 0:
                if idx > 0:
                    context.append((center, self.tokens[idx-1]))
                else:
                    context.append((center, self.tokens[idx+1]))
            context_idx = torch.randint(0, len(context), (1,)).item()
            return (
                torch.tensor([context[context_idx][0]]),
                torch.tensor([context[context_idx][1]])
            )
        else:
            ipt = self.tokens[idx]
            prv = self.tokens[idx-2:idx]
            nex = self.tokens[idx+1:idx+3]
            if len(prv) < 2:
                prv = [0] * (2 - len(prv)) + prv
            if len(nex) < 2:
                nex = nex + [0] * (2 - len(nex))
            return torch.tensor(prv + nex), torch.tensor([ipt])


class MSMARCOTripletDataset(Dataset):
    def __init__(
        self,
        df,
        max_query_len=20,
        max_doc_len=200,
        max_neg_samples=5
    ):
        self.vocab_to_int = pickle.load(
            open(config.VOCAB_TO_ID_PATH, 'rb'))
        self.triplets = []

        # Extract data from DataFrame
        queries = df['queries']
        documents = df['documents']
        labels = df['labels']

        # Process lists of lists structure
        for i in range(len(queries)):
            query = queries[i]
            # List of documents for this query
            docs_list = documents[i].tolist()
            # List of labels for these documents
            labels_list = labels[i].tolist()

            # Skip if we don't have exactly one positive document
            if sum(labels_list) != 1:
                continue

            # Find positive and negative documents
            pos_idx = labels_list.index(1)
            neg_indices = [j for j in range(
                len(labels_list)) if labels_list[j] == 0]

            # Skip if no negatives
            if not neg_indices:
                continue

            pos_doc = docs_list[pos_idx]

            # Limit number of negative samples per positive example
            neg_samples = min(max_neg_samples, len(neg_indices))
            for j in range(neg_samples):
                neg_doc = docs_list[neg_indices[j]]
                self.triplets.append((query, pos_doc, neg_doc))

        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        query, pos_doc, neg_doc = self.triplets[idx]

        query_ids = self._tokenise(query, self.max_query_len)
        pos_doc_ids = self._tokenise(pos_doc, self.max_doc_len)
        neg_doc_ids = self._tokenise(neg_doc, self.max_doc_len)

        return {
            'query_ids': query_ids,
            'pos_doc_ids': pos_doc_ids,
            'neg_doc_ids': neg_doc_ids
        }

    def _tokenise(self, text, max_len):
        tokens = preprocess(str(text))
        # Use 0 for unknown tokens
        ids = [self.vocab_to_int.get(token, 0) for token in tokens[:max_len]]
        # Pad sequence
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return torch.tensor(ids)
