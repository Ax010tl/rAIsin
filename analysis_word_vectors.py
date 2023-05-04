from textblob import TextBlob, Word
import numpy as np
import gensim.downloader as api
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from statistics import mean


# Use the pre-trained model for word2vec
model = api.load("glove-wiki-gigaword-100")


def get_word_vector_from_text(text_blob):
    """
    Gets the average vector from a text blob
    """
    # Store word vectors
    word_vectors = []
    # Iterate over all the words
    for word in np.array(text_blob.words):
        # Lemmatize the word
        w = Word(word).lemma.lower()
        # Check if stopword or not in model
        if w in STOPWORDS or w not in model:
            continue
        # Add the vector
        word_vectors.append(model.get_vector(w))
    # Get average
    vectors_avg = np.mean(word_vectors, axis=0) if len(word_vectors) > 0 else np.zeros(100)
    return vectors_avg.tolist()

def get_sentence_vectors_from_text(text_blob):
    return [get_word_vector_from_text(sentence) for sentence in text_blob.sentences]

def get_vector_cosine_similarity(vector_a, vector_b):
    try:
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return np.dot(vector_a, vector_b) / (norm_a * norm_b)
    except:
        print("Error in get_vector_cosine_similarity")
        return 0