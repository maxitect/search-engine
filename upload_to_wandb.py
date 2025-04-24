import wandb
import os
from tqdm import tqdm

def upload_directory_to_wandb(directory_path, artifact_name, artifact_type):
    """Upload a directory to Weights & Biases as an artifact."""
    # Initialize wandb run
    run = wandb.init(project="search-engine", job_type="upload")
    
    # Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type
    )
    
    # Add files to artifact
    print(f"Uploading {directory_path}...")
    for root, dirs, files in os.walk(directory_path):
        for file in tqdm(files, desc=f"Adding files from {root}"):
            file_path = os.path.join(root, file)
            artifact.add_file(file_path, name=os.path.relpath(file_path, directory_path))
    
    # Upload artifact
    print(f"Uploading {artifact_name}...")
    run.log_artifact(artifact)
    run.finish()
    print(f"Uploaded {artifact_name} successfully!")

def main():
    # Upload data directory
    upload_directory_to_wandb(
        directory_path="data",
        artifact_name="msmarco-data",
        artifact_type="dataset"
    )
    
    # Upload models directory
    upload_directory_to_wandb(
        directory_path="models",
        artifact_name="search-models",
        artifact_type="model"
    )
    
    # Upload text8 embeddings separately due to size
    upload_directory_to_wandb(
        directory_path="models/text8_embeddings",
        artifact_name="text8-embeddings",
        artifact_type="embeddings"
    )

if __name__ == "__main__":
    main() 