import wandb
import os

# Initialize wandb
api = wandb.Api()

# List all runs in your project
runs = api.runs("olliecumming3-machine-learning-institute")

# Print all artifacts
print("\nAvailable Artifacts:")
for run in runs:
    print(f"\nRun: {run.name}")
    for artifact in run.logged_artifacts():
        print(f"  - {artifact.name}")
        # Download artifact
        artifact.download(root=os.path.join("downloaded_models", artifact.name))
        print(f"    Downloaded to: downloaded_models/{artifact.name}") 