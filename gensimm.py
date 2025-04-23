# import gensim.downloader as api
# wv = api.load('word2vec-google-news-300')
from engine.text.gensim_w2v import GensimWord2Vec
# class GensimWord2Vec:
#     def __init__(self, wv):
#         self.wv = wv

#     def get_embedding(self, word):
#         return self.wv.vectors[self.wv.index_to_key.index(word)]


# for index, word in enumerate(wv.index_to_key):
#     if index == 10:
#         break
#     print(f"word #{index}/{len(wv.index_to_key)} is {word}")
#     print(wv.vectors[index])

# print(wv.most_similar(positive=['car', 'minivan'], topn=5))

gensim_w2v = GensimWord2Vec()

print(gensim_w2v.get_word_embedding('car'))
print(gensim_w2v.get_sentence_embeddings('hello my name is john'))
print('')