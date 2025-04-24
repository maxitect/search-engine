# Pull models from HF
from engine.text import setup_language_models, MeanPooledWordEmbedder
from engine.model import Encoder
import torch
from engine.utils import get_wandb_checkpoint_path, get_device
from engine.text.gensim_w2v import GensimWord2Vec
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_semantics_embedder(config: str):
    device = get_device()

    if config == 'my_embeddings' or config == 'gensim_embeddings':
        D_in = 300
        D_hidden = 100
        D_out = 100

        query_encoder = Encoder(
            input_dim=D_in,
            hidden_dim=D_hidden,
            output_dim=D_out,
        )
        doc_encoder = Encoder(
            input_dim=D_in,
            hidden_dim=D_hidden,
            output_dim=D_out,
        )
        if config == 'my_embeddings':
            embeddings = 'self-trained'
            # Pull model from wandb
            checkpoint_path = get_wandb_checkpoint_path(
                'kwokkenton-individual/mlx-week2-search-engine/towers_mlp:v38',
            )
        elif config == 'gensim_embeddings':
            embeddings = 'word2vec-google-news-300'
            # Pull model from wandb
            checkpoint_path = get_wandb_checkpoint_path(
                'kwokkenton-individual/mlx-week2-search-engine/towers_mlp:v44',
            )
            # Load the model
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True,
        )
        query_encoder.load_state_dict(checkpoint['query_encoder_state_dict'])
        doc_encoder.load_state_dict(checkpoint['doc_encoder_state_dict'])

        semantics_embedder = SemanticsEmbedder(
            query_encoder,
            doc_encoder,
            embeddings,
            device,
        )
    elif config == 'baseline':
        baseline_embedder = BaselineSemanticsEmbedder()
        return baseline_embedder
    else:
        raise ValueError(f'Invalid config: {config}')

    return semantics_embedder


class SemanticsEmbedder:
    def __init__(
        self,
        query_encoder: Encoder,
        doc_encoder: Encoder,
        embeddings: str,
        device: torch.device,
    ):
        logger.info(
            f'Initialising SemanticsEmbedder with embeddings: {embeddings}',
        )
        if embeddings not in ['self-trained', 'word2vec-google-news-300']:
            raise ValueError(f'Invalid embeddings: {embeddings}')
        if embeddings == 'self-trained':
            self.tokeniser, self.w2v_model = setup_language_models()

            self.sentence_embedder = MeanPooledWordEmbedder(
                self.tokeniser,
                self.w2v_model,
                device,
            )
            self.embed_fn = self.sentence_embedder.embed_string
        elif embeddings == 'word2vec-google-news-300':
            self.gensim_w2v = GensimWord2Vec()
            self.embed_fn = self.gensim_w2v.get_mean_embedding

        self.query_encoder = query_encoder.to(device)
        self.doc_encoder = doc_encoder.to(device)
        self.device = device

    def embed_query(self, query: str):
        return self._normalize(
            self.query_encoder.forward(self.embed_fn(query).to(self.device)),
        )

    def embed_doc(self, doc: str):
        return self._normalize(
            self.doc_encoder.forward(self.embed_fn(doc).to(self.device)),
        )

    @staticmethod
    def _normalize(vector: torch.tensor) -> torch.tensor:
        """Normalizes a vector to unit length using L2 norm."""
        norm = torch.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class BaselineSemanticsEmbedder:
    def __init__(self):
        logger.info('Initialising BaselineSemanticsEmbedder')
        self.gensim_w2v = GensimWord2Vec()
        self.embed_fn = self.gensim_w2v.get_mean_embedding

    def embed_query(self, query: str):
        return self._normalize(self.embed_fn(query))

    def embed_doc(self, doc: str):
        return self._normalize(
            self.embed_fn(doc),
        )

    @staticmethod
    def _normalize(vector: torch.tensor) -> torch.tensor:
        """Normalizes a vector to unit length using L2 norm."""
        norm = torch.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


if __name__ == '__main__':

    semantics_embedder = setup_semantics_embedder()
    # Test the model
    query = 'What is the capital of France?'
    doc = 'France is a country in Western Europe.'
    query_embedding = semantics_embedder.embed_query(query)
    doc_embedding = semantics_embedder.embed_doc(doc)
    # checkpoint = torch.load(checkpoint_path, map_location=device)
