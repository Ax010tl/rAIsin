from textblob import TextBlob

"""
Given a document blob, return the n-grams of the document
"""
def n_grams_get(document_blob, n=3):
    n_grams = document_blob.ngrams(n=n)
    return n_grams

"""
Given an n-gram and a list of n-grams, return the frequency of the n-gram in the list
"""
def n_grams_frequency(n_gram, n_grams):
    return n_grams.count(n_gram)

"""
Using the formula nd_1(A, B) from https://ceur-ws.org/Vol-502/paper8.pdf,
calculate the distance between two n-grams.
"""
# Get the distance between two n-grams
def n_grams_distance(base_n_grams, document_n_grams):
    distance = 0
    # For each n-gram in the base corpus
    for g in base_n_grams:
        # Get the frequency of the n-gram in the base corpus
        frequency_base = n_grams_frequency(g, base_n_grams)
        # Get the frequency of the n-gram in the document corpus
        frequency_corpus = n_grams_frequency(g, document_n_grams)
        # Calculate the distance
        distance += pow(2 * (frequency_base - frequency_corpus) / (frequency_base + frequency_corpus), 2)
    # Return the average distance
    return distance / (4 * len(base_n_grams))

"""
Given a document blob, return the average distance between the n-grams of the document and the overall document
"""
def n_grams_analyze(document_blob):
    score = 0
    # Get n_grams from the corpus
    n_grams = n_grams_get(document_blob)
    for g in n_grams:
        # Get the distance between the n_gram and the n_grams of the corpus
        distance = n_grams_distance([g], n_grams)
        # Add the distance to the score
        score += distance
    # Return the average distance
    return score / len(n_grams)