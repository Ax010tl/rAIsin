from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def get_word_frequency_matrix(document_texts):
    # Create a vectorizer that only accepts words
    vectorizer = CountVectorizer(token_pattern=r"\b[a-zA-Z]+\b")
    # Get word frequency matrix
    X = vectorizer.fit_transform(document_texts)
    # Convert to a dataframe
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names_out())
    return df