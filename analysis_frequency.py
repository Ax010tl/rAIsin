from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Get word frequency matrix from a list of texts
def get_word_frequency_matrix(document_texts):
    # Create a vectorizer that only accepts words
    vectorizer = CountVectorizer(token_pattern=r"\b[a-zA-Z]+\b")
    # Get word frequency matrix
    X = vectorizer.fit_transform(document_texts)
    # Convert to a dataframe
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())
    return df

# Receive a word frequency matrix and a word frequency dataframe (a single row) and return the max cosine similarity
def get_word_frequency_matrix_max_cosine_similarity(word_frequency_matrix_df, word_frequency_df):
    # Add both dataframes together
    df = pd.concat([word_frequency_matrix_df, word_frequency_df], axis=0)
    df_filled = df.fillna(0)
    # Select all but the last row
    word_frequency_matrix_df = df_filled.iloc[:-1]
    # Select the last row
    word_frequency = df_filled.iloc[-1, :].to_numpy()
    # Store max cosine similarity
    max_cosine_similarity = 0
    # Iterate over all rows
    for index, row in word_frequency_matrix_df.iterrows():
        numpy_row = row.to_numpy()
        # Get cosine similarity 
        cosine_similarity = np.dot(numpy_row, word_frequency) / (np.linalg.norm(numpy_row) * np.linalg.norm(word_frequency))
        # Check if max
        if cosine_similarity > max_cosine_similarity:
            max_cosine_similarity = cosine_similarity
    return max_cosine_similarity
