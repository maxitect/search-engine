import wandb
import os
from tqdm import tqdm
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_directory_to_wandb(directory_path, artifact_name, artifact_type):
    """Upload a directory to Weights & Biases as an artifact."""
    try:
        # Initialize wandb run
        run = wandb.init(project="search-engine", job_type="upload")
        
        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=f"Upload of {directory_path} directory"
        )
        
        # Add files to artifact
        logger.info(f"Starting upload of {directory_path}...")
        total_size = 0
        file_count = 0
        
        for root, dirs, files in os.walk(directory_path):
            for file in tqdm(files, desc=f"Adding files from {root}"):
                try:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    file_count += 1
                    
                    # Log large files
                    if file_size > 100 * 1024 * 1024:  # Files larger than 100MB
                        logger.info(f"Adding large file: {file} ({file_size / (1024*1024):.2f} MB)")
                    
                    artifact.add_file(file_path, name=os.path.relpath(file_path, directory_path))
                except Exception as e:
                    logger.error(f"Error adding file {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Total files: {file_count}, Total size: {total_size / (1024*1024):.2f} MB")
        
        # Upload artifact
        logger.info(f"Uploading {artifact_name}...")
        run.log_artifact(artifact)
        run.finish()
        logger.info(f"Successfully uploaded {artifact_name}!")
        
    except Exception as e:
        logger.error(f"Error uploading {artifact_name}: {str(e)}")
        raise

def main():
    try:
        # Upload data directory
        logger.info("Starting data upload...")
        upload_directory_to_wandb(
            directory_path="data",
            artifact_name="msmarco-data",
            artifact_type="dataset"
        )
        
        # Upload models directory
        logger.info("Starting models upload...")
        upload_directory_to_wandb(
            directory_path="models",
            artifact_name="search-models",
            artifact_type="model"
        )
        
        # Upload text8 embeddings separately due to size
        logger.info("Starting embeddings upload...")
        upload_directory_to_wandb(
            directory_path="models/text8_embeddings",
            artifact_name="text8-embeddings",
            artifact_type="embeddings"
        )
        
    except Exception as e:
        logger.error(f"Error in main upload process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 