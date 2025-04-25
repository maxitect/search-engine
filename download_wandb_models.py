import wandb
import os
import shutil
from tqdm import tqdm

# Initialize wandb
api = wandb.Api()

# Define artifacts to download
artifacts = [
    {
        "path": "olliecumming3-machine-learning-institute/search-engine/msmarco-data:v0",
        "dest": "data"
    },
    {
        "path": "olliecumming3-machine-learning-institute/search-engine/search-models:v0",
        "dest": "models"
    },
    {
        "path": "olliecumming3-machine-learning-institute/search-engine/text8-embeddings:v0",
        "dest": "models/text8_embeddings"
    }
]

def download_artifact(artifact_path, destination):
    """Download a single artifact to the specified destination."""
    print(f"\nDownloading artifact: {artifact_path}")
    try:
        # Create destination directory if it doesn't exist
        os.makedirs(destination, exist_ok=True)
        
        # Download the artifact
        artifact = api.artifact(artifact_path)
        artifact.download(root=destination)
        
        # Print downloaded files
        print(f"\nDownloaded files to {destination}:")
        for root, dirs, files in os.walk(destination):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"- {file_path} ({file_size / (1024*1024):.2f} MB)")
                
    except Exception as e:
        print(f"Error downloading artifact {artifact_path}: {e}")
        return False
    return True

def main():
    # Download each artifact
    for artifact in artifacts:
        success = download_artifact(artifact["path"], artifact["dest"])
        if not success:
            print(f"Failed to download {artifact['path']}")
            continue

if __name__ == "__main__":
    main() 