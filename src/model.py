import torch


class SkipGram(torch.nn.Module):
    def __init__(self, voc, emb):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(
            in_features=emb, out_features=voc, bias=False)

    def forward(self, inpt):
        emb = self.emb(inpt)
        out = self.ffw(emb.squeeze(1))
        return out


class CBOW(torch.nn.Module):
    def __init__(self, voc, emb):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(
            in_features=emb, out_features=voc, bias=False)

    def forward(self, inpt):
        emb = self.emb(inpt)
        emb = emb.mean(dim=1)
        out = self.ffw(emb)
        return out


class Regressor(torch.nn.Module):
    def __init__(self, emb_dim=128, domain_size=1000, time_features=2):
        super().__init__()
        self.domain_emb = torch.nn.Embedding(domain_size, 16)
        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_features=emb_dim + 16 +
                            time_features + 1, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=16),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, title_emb, domain_idx, time_feats, title_length):
        domain_emb = self.domain_emb(domain_idx)
        combined = torch.cat(
            [title_emb, domain_emb, time_feats, title_length.unsqueeze(1)],
            dim=1
        )
        out = self.seq(combined)
        return out
