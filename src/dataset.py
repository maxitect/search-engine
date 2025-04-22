import numpy as np
import src.config as config
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

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


class MSMARCODataset(Dataset):
    def __init__(
            self,
            queries,
            documents,
            labels,
            vocab_to_int,
            max_query_len=20,
            max_doc_len=200
    ):
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.vocab_to_int = vocab_to_int
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        docs = self.documents[idx]  # This is now an array of documents
        label = self.labels[idx]  # This could be an array of labels too

        # Ensure docs is a list/array
        if not isinstance(docs, (list, tuple, np.ndarray)):
            docs = [docs]

        # Ensure labels match documents
        if isinstance(
            label,
            (list,
             tuple,
             np.ndarray)
        ) and len(label) == len(docs):
            labels = label
        else:
            # If we have a single label,
            # apply it to all docs or use the first label
            labels = [label] * len(docs) if not isinstance(
                label,
                (list, tuple,
                 np.ndarray)
            ) else [label[0]] * len(docs)

        # Tokenize query
        query_ids = self._tokenise(query, self.max_query_len)

        # Tokenize all documents
        doc_ids_list = []
        for doc in docs:
            doc_ids = self._tokenise(doc, self.max_doc_len)
            doc_ids_list.append(doc_ids)

        return {
            'query_ids': query_ids,
            'doc_ids_list': doc_ids_list,
            'labels': labels
        }

    def _tokenise(self, text, max_len):
        tokens = preprocess(text)
        # For unknown tokens, use 0 (PAD token) instead of raising an error
        ids = [self.vocab_to_int.get(token, 0) for token in tokens[:max_len]]
        # Pad sequence
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return torch.tensor(ids)


def custom_collate_fn(batch):
    batch_dict = {
        'query_ids': [],
        'doc_ids': [],
        'label': []
    }

    for sample in batch:
        # Each sample has query_ids, doc_ids (possibly a list), and label
        # (possibly a list)
        batch_dict['query_ids'].append(sample['query_ids'])

        # If doc_ids is a list of tensors, we'll process each one separately
        if isinstance(sample['doc_ids'], list):
            for doc, lab in zip(sample['doc_ids'], sample['label']):
                batch_dict['query_ids'].append(
                    sample['query_ids'])  # Repeat the query
                batch_dict['doc_ids'].append(doc)
                batch_dict['label'].append(lab)
        else:
            # Just a single document
            batch_dict['doc_ids'].append(sample['doc_ids'])
            batch_dict['label'].append(sample['label'])

    # Stack the tensors
    result = {
        'query_ids': torch.stack(batch_dict['query_ids']),
        'doc_ids': torch.stack(batch_dict['doc_ids']),
        'label': torch.tensor(batch_dict['label'])
    }

    return result


def generate_triplets(dataset, batch_size):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    triplets = []

    for batch in dataloader:
        queries = batch['query_ids']
        docs = batch['doc_ids']
        labels = batch['label']

        # For each query, find a positive and negative document
        for i in range(len(queries)):
            query = queries[i]
            pos_indices = [j for j in range(len(labels)) if labels[j] == 1]
            neg_indices = [j for j in range(len(labels)) if labels[j] == 0]

            if pos_indices and neg_indices:
                pos_idx = pos_indices[0]
                neg_idx = neg_indices[0]
                triplets.append((query, docs[pos_idx], docs[neg_idx]))

    return triplets
