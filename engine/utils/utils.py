import glob
import os

import wandb


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