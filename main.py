from engine.text import setup_language_models
from engine.data import MSMarcoDataset


if __name__ == '__main__':
    tokeniser, w2v_model = setup_language_models()

    print(tokeniser.tokenise_string('hello world... 333'))

    train_ds = MSMarcoDataset('train')
    query, pos_docs, neg_docs = train_ds[0]
    print(query)
    print(pos_docs)
    print(neg_docs)

    print(tokeniser.tokenise_string(query))
    print(tokeniser.tokenise_string(pos_docs[0]))
    print(tokeniser.tokenise_string(neg_docs[0]))

    print(tokeniser.tokens_to_words(tokeniser.tokenise_string(query)))
