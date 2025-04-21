# search-engine

A search engine which takes in queries and produces a list of relevant documents using two tower architecture, trained on MS MARCO.

## Deliverable

- Inputs: a query, of any length
- Outputs: the most similar document

## Training

In training

- Inputs: a query sentence
- Outputs: the id of the most similar document

### Loss function

Hinge loss

- Want high loss when positive distances are large, and when negative distances are small
- Want no loss if the distances are 'far enough' by a margin

## MS Marco Dataset

Available on HuggingFace.

## References

Karpukhin, Vladimir, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. ‘Dense Passage Retrieval for Open-Domain Question Answering’. arXiv, 30 September 2020. https://doi.org/10.48550/arXiv.2004.04906.
