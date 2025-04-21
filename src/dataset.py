import src.config as config
import torch
import pickle
import datetime


class Wiki(torch.utils.data.Dataset):
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


class HackerNewsDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            titles,
            domains,
            timestamps,
            scores,
            vocab_to_int,
            max_title_len=100
    ):
        self.titles = titles
        self.domains = domains
        self.timestamps = timestamps
        self.scores = scores
        self.vocab_to_int = vocab_to_int
        self.max_title_len = max_title_len

        self.domain_to_idx = {d: i+1 for i, d in enumerate(set(domains))}
        self.domain_to_idx['<UNK>'] = 0

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        domain = self.domains[idx]
        timestamp = self.timestamps[idx]
        score = self.scores[idx]

        title_tokens = title.lower().split()
        title_ids = []
        for word in title_tokens[:self.max_title_len]:
            if word in self.vocab_to_int:
                title_ids.append(self.vocab_to_int[word])
            else:
                title_ids.append(0)

        if len(title_ids) < self.max_title_len:
            title_ids.extend([0] * (self.max_title_len - len(title_ids)))

        domain_idx = self.domain_to_idx.get(domain, 0)

        dt = datetime.datetime.fromisoformat(timestamp)
        day_of_week = dt.weekday()
        hour_of_day = dt.hour

        time_feats = torch.tensor(
            [day_of_week/6, hour_of_day/23], dtype=torch.float32)
        title_length = torch.tensor(
            min(len(title), 200)/200, dtype=torch.float32)

        return {
            'title_ids': torch.tensor(title_ids),
            'domain_idx': torch.tensor(domain_idx),
            'time_feats': time_feats,
            'title_length': title_length,
            'score': torch.tensor(score, dtype=torch.float32)
        }
