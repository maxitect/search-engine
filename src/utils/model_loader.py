import os
import shutil
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
    checkpoint = torch.load(
        map_location=dev,
        f=config.SKIPGRAM_BEST_MODEL_PATH
    )

    if 'in_embed.weight' in checkpoint:
        skipgram.load_state_dict(checkpoint)
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        skipgram.load_state_dict(checkpoint['state_dict'])
    else:
        print("Invalid SkipGram checkpoint format")
        raise RuntimeError("Invalid SkipGram checkpoint format")

    skipgram.to(dev)
    skipgram.eval()
    print(f'Loaded SkipGram model from {config.SKIPGRAM_BEST_MODEL_PATH}')
    return skipgram


def load_twotowers(vocab_to_int):
    """Load the two-tower model and vocabulary mapping."""
    skipgram = load_skipgram(vocab_to_int)
    # initialise towers
    print('Loading Two Towers model...')
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
    checkpoint = torch.load(
        map_location=dev,
        f=config.TWOTOWERS_BEST_MODEL_PATH
    )
    # load pretrained weights
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f'Loaded Two Towers model from {config.TWOTOWERS_BEST_MODEL_PATH}')
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

        artifact_dir = artifact.download(root=config.SKIPGRAM_CHECKPOINT_DIR)
        pth_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
        original_path = os.path.join(artifact_dir, pth_files[0])
        custom_path = os.path.join(dir, 'best_model.pth')
        if os.path.exists(custom_path) and os.path.samefile(
            original_path,
            custom_path
        ):
            print("Best model already in directory, using local file.")
        else:
            shutil.copy(original_path, custom_path)
