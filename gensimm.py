
from engine.text.gensim_w2v import GensimWord2Vec
from engine.data.ms_marco import load_ms_marco

gensim_w2v = GensimWord2Vec()


ds = load_ms_marco()

split = 'train'
for i in range(10):
    row = ds[split][i]
    query_embedded = gensim_w2v.get_sentence_embeddings(row['query'])
    row['query_embedded'] = query_embedded
    break