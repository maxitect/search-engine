import src.config as config
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

from utils.tokenise import preprocess


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
        doc = self.documents[idx]
        label = self.labels[idx]

        # Tokenise and convert to IDs
        query_ids = self._tokenise(query, self.max_query_len)
        doc_ids = self._tokenise(doc, self.max_doc_len)

        return {
            'query_ids': query_ids,
            'doc_ids': doc_ids,
            'label': label
        }

    def _tokenise(self, text, max_len):
        tokens = preprocess(text)
        ids = [self.vocab_to_int.get(token, self.vocab_to_int['<UNK>'])
               for token in tokens[:max_len]]
        # Pad sequence
        if len(ids) < max_len:
            ids = ids + [self.vocab_to_int['<PAD>']] * (max_len - len(ids))
        return torch.tensor(ids)


def generate_triplets(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
