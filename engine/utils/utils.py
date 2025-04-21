import glob
import os

import wandb
import torch


def get_wandb_checkpoint_path(artifact_name: str, run=None):
    if run is None:
        api = wandb.Api()
        artifact = api.artifact(artifact_name)
    else:
        artifact = run.use_artifact(artifact_name)

    artifact_dir = artifact.download()
    # Get the first .pth file in the directory
    checkpoint_path = glob.glob(os.path.join(artifact_dir, '*.pth'))[0]
    return checkpoint_path


def get_device():
    # Training setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS (GPU) backend.')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA (GPU) backend.')
    else:
        device = torch.device('cpu')
        print('Using CPU backend.')
    return device
