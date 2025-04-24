import wandb
import os
import shutil

# Initialize wandb
api = wandb.Api()

# Specify the artifact path
artifact_path = "olliecumming3-machine-learning-institute/two-tower-models/all_models:v0"

# Create or clean models directory
if os.path.exists("models"):
    shutil.rmtree("models")
os.makedirs("models", exist_ok=True)

# Download the artifact
print(f"Downloading artifact: {artifact_path}")
try:
    artifact = api.artifact(artifact_path)
    artifact.download(root="models")
except Exception as e:
    print(f"Error downloading artifact: {e}")
    exit(1)

# Print downloaded files
print("\nDownloaded files:")
for root, dirs, files in os.walk("models"):
    for file in files:
        print(f"- {os.path.join(root, file)}") 