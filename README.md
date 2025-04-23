# search-engine

A search engine which takes in queries and produces a list of relevant documents using two tower architecture, trained on MS MARCO.

Semantic search is a data searching technique that focuses on understanding the contextual meaning and intent behind a user’s search query, rather than only matching keywords.

## Deliverable

- Inputs: a query, of any length
- Outputs: the $k$ ids of the most similar document

## Training

In training

- Inputs: a query sentence
- Outputs: the id of the most similar document

### Loss function

Hinge loss

- Want high loss when positive distances are large, and when negative distances are small
- Want no loss if the distances are 'far enough' by a margin

Loss function
$$L = \frac{1}{B}\sum_{B} \big [ \max(\text{dist}(q,p^{+}) - \frac{1}{K} \sum_K \text{dist}(q,p^{-}_k) , 0\big)]$$

## MS Marco Dataset

Available on HuggingFace.

## Potential pitfalls

1. Word not included in the tokeniser (not in Wikipedia/ or niche words), this is especially pertinent for acronyms. Example, we had a query called 'what is rba', which turned into '['what', 'is', '<UNK>']' in tokenised form.

### Notes
1. I trained two models with either higher


## Deployment

```bash
python3 create_db.py

```


## Installation

Conda

```bash
# Install miniconda for the system
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate

# Create conda environment
conda env create --file environment.yml

# Updates conda
conda update -n base -c defaults conda

# Update environment
conda activate ss-env

# Login manually
wandb login
huggingface-cli login

# Add Git credentials
git config --global user.name ""
git config --global user.email ""
```

### Todo:

Precompute all word vectors using the skipgram model and save as a dictionary

https://www.anaconda.com/docs/getting-started/miniconda/install#linux

Setup your HF tokens

## References

Karpukhin, Vladimir, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. ‘Dense Passage Retrieval for Open-Domain Question Answering’. arXiv, 30 September 2020. https://doi.org/10.48550/arXiv.2004.04906.
