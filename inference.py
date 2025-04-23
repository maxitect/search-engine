# Pull models from HF
from engine.text import setup_language_models, MeanPooledWordEmbedder
from engine.model import Encoder
import torch
from engine.utils import get_wandb_checkpoint_path, get_device


def setup_semantics_embedder():
    # Pull model from wandb
    device = get_device()
    D_in = 300
    D_hidden = 100
    D_out = 100

    checkpoint_path = get_wandb_checkpoint_path(
        'kwokkenton-individual/mlx-week2-search-engine/towers_mlp:v38',
    )
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
    # Load the model
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True,
    )
    query_encoder.load_state_dict(checkpoint['query_encoder_state_dict'])
    doc_encoder.load_state_dict(checkpoint['doc_encoder_state_dict'])

    semantics_embedder = SemanticsEmbedder(
        query_encoder,
        doc_encoder,
        device,
    )
    return semantics_embedder


class SemanticsEmbedder:
    def __init__(
        self, query_encoder: Encoder,
        doc_encoder: Encoder,
        device: torch.device,
    ):
        self.tokeniser, self.w2v_model = setup_language_models()

        self.sentence_embedder = MeanPooledWordEmbedder(
            self.tokeniser,
            self.w2v_model,
            device,
        )
        self.embed_fn = self.sentence_embedder.embed_string
        self.query_encoder = query_encoder.to(device)
        self.doc_encoder = doc_encoder.to(device)
        self.device = device

    def embed_query(self, query: str):
        return self._normalize(
            self.query_encoder.forward(self.embed_fn(query)),
        )

    def embed_doc(self, doc: str):
        return self._normalize(
            self.doc_encoder.forward(self.embed_fn(doc)),
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
