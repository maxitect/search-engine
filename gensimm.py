
from engine.text.gensim_w2v import GensimWord2Vec
from engine.data.ms_marco import load_ms_marco
import logging
from huggingface_hub import HfApi
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)






def add_embeddings_to_dataset(ds, gensim_w2v):
    for split in ['test', 'train', 'validation', ]:
        logger.info(f"Processing {split} split...")
        split_ds = ds[split]
        
        # Add embeddings to each example
        def process_example(example):
            # Get query embedding
            query_embedding = gensim_w2v.get_sentence_embeddings(example['query']).numpy()
            
            # Get passage embeddings
            passage_embeddings = [
                gensim_w2v.get_sentence_embeddings(passage).numpy()
                for passage in example['passages']['passage_text']
            ]
            
            # Add embeddings to the example
            example['query_embedding'] = query_embedding
            example['passage_embeddings'] = passage_embeddings
            
            return example
        
        # Process the dataset
        processed_ds = split_ds.map(
            process_example,
            batched=False,
            desc=f"Adding embeddings to {split} split"
        )
        
        # Update the dataset
        ds[split] = processed_ds

    return ds

# def upload_to_huggingface(dataset, repo_name: str, organization: str = None):
#     """
#     Upload the dataset to HuggingFace Hub.
    
#     Args:
#         dataset: The processed dataset
#         repo_name: Name of the repository to create
#         organization: Optional organization name
#     """
#     # Create repository
#     repo_id = f"{organization}/{repo_name}" if organization else repo_name
#     create_repo(repo_id, repo_type="dataset", exist_ok=True)
    
#     # Upload the dataset
#     logger.info(f"Uploading dataset to {repo_id}...")
#     dataset.push_to_hub(repo_id)
#     logger.info("Upload complete!")

def main():
    gensim_w2v = GensimWord2Vec()
    data_path = 'data/ms_marco_with_embeddings.hf'
    # # Add embeddings to the dataset
    # logger.info("Adding embeddings to MS MARCO dataset...")
    ds = load_ms_marco()
    processed_ds = add_embeddings_to_dataset(ds, gensim_w2v)
    processed_ds.save_to_disk(data_path)
    print(processed_ds)

    api = HfApi()
    api.upload_folder(
        folder_path=data_path,
        repo_id='kwokkenton/ms_marco',
        repo_type='dataset',
    )

    # print('')
    # Upload to HuggingFace
    # repo_name = "ms_marco_with_embeddings"
    # organization = "your_organization"  # Replace with your organization name or None
    # upload_to_huggingface(processed_ds, repo_name, organization)

if __name__ == "__main__":
    main()
    # from datasets import load_dataset

    # from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk("data/ms_marco_with_embeddings")
    # print(dataset)