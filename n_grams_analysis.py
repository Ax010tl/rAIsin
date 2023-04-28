from textblob import TextBlob

# Get n_grams from a corpus
def n_grams_get(corpus_blob, n=3):
    n_grams = corpus_blob.ngrams(n=n)
    return n_grams

# Get the frequency of a n-gram in a corpus
def n_grams_frequency(n_gram, n_grams):
    return n_grams.count(n_gram)

# Get the distance between two n-grams
def n_grams_distance(base_n_grams, corpus_n_grams):
    distance = 0
    # For each n-gram in the base corpus
    for g in base_n_grams:
        # Get the frequency of the n-gram in the base corpus
        frequency_base = n_grams_frequency(g, base_n_grams)
        # Get the frequency of the n-gram in the corpus
        frequency_corpus = n_grams_frequency(g, corpus_n_grams)
        # Calculate the distance
        distance += pow(2 * (frequency_base - frequency_corpus) / (frequency_base + frequency_corpus), 2)
    # Return the average distance
    return distance / (4 * len(base_n_grams))

# Calculate style change function
def n_grams_analyze(corpus_blob):
    score = 0
    # Get n_grams from the corpus
    n_grams = n_grams_get(corpus_blob)
    for g in n_grams:
        # Get the distance between the n_gram and the n_grams of the corpus
        distance = n_grams_distance([g], n_grams)
        # Add the distance to the score
        score += distance
    # Return the average distance
    return score / len(n_grams)