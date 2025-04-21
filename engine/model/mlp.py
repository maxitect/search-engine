import torch


class BagOfWordsMLP(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_words: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_words)
