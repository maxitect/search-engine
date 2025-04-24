import torch
import wandb

import src.config as config
from src.models.skipgram import SkipGram
from src.models.twotowers import QryTower, DocTower, TwoTowerModel

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_skipgram(vocab_to_int):
    """Load the SkipGram model."""
    print('Loading SkipGram model...')
    skipgram = SkipGram(len(vocab_to_int), config.EMBEDDING_DIM)
    skipgram.load_state_dict(
        torch.load(map_location=dev, f=config.SKIPGRAM_BEST_MODEL_PATH)
    )

    skipgram.to(dev)
    skipgram.eval()
    print(f'Loaded SkipGram model from {config.SKIPGRAM_BEST_MODEL_PATH}')
    return skipgram


def load_twotowers(vocab_to_int):
    """Load the two-tower model with nested checkpoint structure."""
    print('Loading Two Tower model...')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skipgram = load_skipgram(vocab_to_int)

    qry = QryTower(
        vocab_size=len(vocab_to_int),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout_rate=config.DROPOUT_RATE,
        skipgram_model=skipgram
    )

    doc = DocTower(
        vocab_size=len(vocab_to_int),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout_rate=config.DROPOUT_RATE,
        skipgram_model=skipgram
    )

    model = TwoTowerModel(qry, doc)
    model.to(dev)

    checkpoint = torch.load(
        map_location=dev,
        f=config.TWOTOWERS_BEST_MODEL_PATH
    )

    # Create a flattened state dict from the nested structure
    state_dict = {}
    for key, value in checkpoint['query_tower'].items():
        state_dict[f'query_tower.{key}'] = value
    for key, value in checkpoint['doc_tower'].items():
        state_dict[f'doc_tower.{key}'] = value

    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded Two Tower model from {config.TWOTOWERS_BEST_MODEL_PATH}')
    return model


def download_model(download, model_name, dir):
    if download:
        print(
            f"Downloading {model_name} from latest artifact..."
        )

        api = wandb.Api()
        artifact = api.artifact(
            'maxime-downe-founders-and-coders/search-engine/'
            f'{model_name}:latest')

        artifact.download(root=dir)
