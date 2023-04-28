from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
import numpy as np
from numpy.linalg import norm
from textblob import TextBlob

model = api.load("glove-wiki-gigaword-50")# train a model from the corpus

def vector_from_text(text_blob):
    vectors = []
    for word in text_blob.words:
        try:
            vectors.append(model.get_vector(word))
        except:
            pass
    # Return the average vector
    return sum(vectors) / len(vectors)

def vector_similarity_cosine(vec1, vec2):
    return np.dot(vec1, vec2)/(norm(vec1)*norm(vec2))
