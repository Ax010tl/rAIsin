from textblob import TextBlob
import numpy as np
import pyphen
import textstat

pyphen_dic = pyphen.Pyphen(lang="en")

# Get n-grams from text
def get_ngrams_from_text(text_blob, n=3):
    ngrams = text_blob.ngrams(n)
    return ngrams

# Get distance between two ngams
def get_distance_between_ngrams(ngram_list_a, ngram_list_b):
    distance = 0
    # For each ngram in list a
    for g in ngram_list_a:
        # Get frequency of ngram in list a
        freq_a = ngram_list_a.count(g)
        # Get frequency of ngram in list b
        freq_b = ngram_list_b.count(g)
        # Add to distance
        distance += pow((2 * freq_a - freq_b) / (freq_a + freq_b), 2)
    # Return average distance
    return distance / (4 * len(ngram_list_a))

def get_ngrams_score_from_text(text_blob):
    score = 0
    ngrams = get_ngrams_from_text(text_blob)
    for g in ngrams:
        score += get_distance_between_ngrams([g], ngrams)
    return score / len(ngrams)


# Get average sentence length from text blob
def get_average_sentence_length_from_text(text_blob):
    sentence_lengths = [len(sentence.words) for sentence in text_blob.sentences]
    return sum(sentence_lengths) / len(sentence_lengths)

# Get normalized standard deviation of sentence length from text blob
def get_normalized_standard_deviation_of_sentence_length_from_text(text_blob):
    sentence_lengths = [len(sentence.words) for sentence in text_blob.sentences]
    return np.std(sentence_lengths) / np.mean(sentence_lengths)

# Get average word length from text blob
def get_average_word_length_from_text(text_blob):
    word_lengths = [len(word) for word in text_blob.words]
    return sum(word_lengths) / len(word_lengths)

# Get normalized standard deviation of word length from text blob
def get_normalized_standard_deviation_of_word_length_from_text(text_blob):
    word_lengths = [len(word) for word in text_blob.words]
    return np.std(word_lengths) / np.mean(word_lengths)

# Get average syllables per word from text blob
def get_average_syllables_per_word_from_text(text_blob):
    syllables_per_word = [len(pyphen_dic.inserted(word).split("-")) for word in text_blob.words]
    return sum(syllables_per_word) / len(syllables_per_word)

# Get normalized standard deviation of syllables per word from text blob
def get_normalized_standard_deviation_of_syllables_per_word_from_text(text_blob):
    syllables_per_word = [len(pyphen_dic.inserted(word).split("-")) for word in text_blob.words]
    return np.std(syllables_per_word) / np.mean(syllables_per_word)

# Get average Flesch-Kincaid reading ease score per sentence from text blob
def get_average_flesch_kincaid_reading_ease_score_per_sentence_from_text(text_blob):
    flesch_kincaid_reading_ease_scores = [textstat.flesch_reading_ease(sentence.string) for sentence in text_blob.sentences]
    return sum(flesch_kincaid_reading_ease_scores) / len(flesch_kincaid_reading_ease_scores)

# Get normalized standard deviation of Flesch-Kincaid reading ease score per sentence from text blob
def get_normalized_standard_deviation_of_flesch_kincaid_reading_ease_score_per_sentence_from_text(text_blob):
    flesch_kincaid_reading_ease_scores = [textstat.flesch_reading_ease(sentence.string) for sentence in text_blob.sentences]
    return np.std(flesch_kincaid_reading_ease_scores) / np.mean(flesch_kincaid_reading_ease_scores)

# Get average Gunning Fog index per sentence from text blob
def get_average_gunning_fog_index_per_sentence_from_text(text_blob):
    gunning_fog_indices = [textstat.gunning_fog(sentence.string) for sentence in text_blob.sentences]
    return sum(gunning_fog_indices) / len(gunning_fog_indices)

# Get normalized standard deviation of Gunning Fog index per sentence from text blob
def get_normalized_standard_deviation_of_gunning_fog_index_per_sentence_from_text(text_blob):
    gunning_fog_indices = [textstat.gunning_fog(sentence.string) for sentence in text_blob.sentences]
    return np.std(gunning_fog_indices) / np.mean(gunning_fog_indices)

# Calculate stylometry features from text blob
def get_stylometry_dict_from_text(text_blob):
    # Word n-grams
    word_ngrams = get_ngrams_score_from_text(text_blob)
    # Pos n-grams
    blob_pos = " ".join([word[1] for word in text_blob.tags])
    pos_ngrams = get_ngrams_score_from_text(TextBlob(blob_pos))
    # Average sentence length
    average_sentence_length = get_average_sentence_length_from_text(text_blob)
    # Normalized standard deviation of sentence length
    normalized_standard_deviation_of_sentence_length = get_normalized_standard_deviation_of_sentence_length_from_text(text_blob)
    # Average word length
    average_word_length = get_average_word_length_from_text(text_blob)
    # Normalized standard deviation of word length
    normalized_standard_deviation_of_word_length = get_normalized_standard_deviation_of_word_length_from_text(text_blob)
    # Average syllables per word
    average_syllables_per_word = get_average_syllables_per_word_from_text(text_blob)
    # Normalized standard deviation of syllables per word
    normalized_standard_deviation_of_syllables_per_word = get_normalized_standard_deviation_of_syllables_per_word_from_text(text_blob)
    # Average Flesch-Kincaid reading ease score per sentence
    average_flesch_kincaid_reading_ease_score_per_sentence = get_average_flesch_kincaid_reading_ease_score_per_sentence_from_text(text_blob)
    # Normalized standard deviation of Flesch-Kincaid reading ease score per sentence
    normalized_standard_deviation_of_flesch_kincaid_reading_ease_score_per_sentence = get_normalized_standard_deviation_of_flesch_kincaid_reading_ease_score_per_sentence_from_text(text_blob)
    # Average Gunning Fog index per sentence
    average_gunning_fog_index_per_sentence = get_average_gunning_fog_index_per_sentence_from_text(text_blob)
    # Normalized standard deviation of Gunning Fog index per sentence
    normalized_standard_deviation_of_gunning_fog_index_per_sentence = get_normalized_standard_deviation_of_gunning_fog_index_per_sentence_from_text(text_blob)
    # Return stylometry dict
    return {
        "word_ngrams": word_ngrams,
        "pos_ngrams": pos_ngrams,
        "average_sentence_length": average_sentence_length,
        "normalized_standard_deviation_of_sentence_length": normalized_standard_deviation_of_sentence_length,
        "average_word_length": average_word_length,
        "normalized_standard_deviation_of_word_length": normalized_standard_deviation_of_word_length,
        "average_syllables_per_word": average_syllables_per_word,
        "normalized_standard_deviation_of_syllables_per_word": normalized_standard_deviation_of_syllables_per_word,
        "average_flesch_kincaid_reading_ease_score_per_sentence": average_flesch_kincaid_reading_ease_score_per_sentence,
        "normalized_standard_deviation_of_flesch_kincaid_reading_ease_score_per_sentence": normalized_standard_deviation_of_flesch_kincaid_reading_ease_score_per_sentence,
        "average_gunning_fog_index_per_sentence": average_gunning_fog_index_per_sentence,
        "normalized_standard_deviation_of_gunning_fog_index_per_sentence": normalized_standard_deviation_of_gunning_fog_index_per_sentence
    }