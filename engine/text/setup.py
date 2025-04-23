from huggingface_hub import hf_hub_download
from engine.utils import get_wandb_checkpoint_path
from .word2vec import Word2Vec, SkipGram

from .tokeniser import get_tokeniser

from ..utils.utils import get_device
import torch


def setup_language_models():
    text8_path = hf_hub_download(
        repo_id='kwokkenton/hn-upvotes', filename='text8.parquet',
        repo_type='dataset',
    )

    # Load the word2vec model
    w2v_checkpoint_path = get_wandb_checkpoint_path(
        'kwokkenton-individual/mlx-week1-word2vec/skipgram:v34',
    )

    tokeniser = get_tokeniser(text8_path)
    # vocab_size = tokeniser.vocab_size
    w2v_model = get_word2vec_from_checkpoint(w2v_checkpoint_path).eval()

    return tokeniser, w2v_model


def get_word2vec_from_checkpoint(checkpoint_path: str):
    """
    Get the word2vec model from the checkpoint
    """
    device = get_device()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True,
    )

    embedding_dim = checkpoint['embedding_dim']
    vocab_size = checkpoint['vocab_size']
    mode = checkpoint['mode']
    if mode == 'cbow':
        model = Word2Vec(vocab_size, embedding_dim, mode='cbow')
    else:
        model = SkipGram(vocab_size, embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


class MeanPooledWordEmbedder:
    def __init__(self, tokeniser, w2v_model, device):
        self.tokeniser = tokeniser
        self.w2v_model = w2v_model
        self.device = device
        self.w2v_model = self.w2v_model.to(device)

    def embed_string(self, q: str):
        """
        Embed a string into a mean-pooled word embedding shape (D,).

        Args:
            q: The string to embed.
            tokeniser: The tokeniser instance.
            w2v_model: The word embedding model instance.

        Returns:
            A tensor of shape (D,).
        """
        with torch.no_grad():
            token = self.tokeniser.tokenise_string(q)
            embedding = self.w2v_model.forward(
                torch.tensor(token).to(self.device),
            ).mean(dim=0)
        return embedding
