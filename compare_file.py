import os
import sys
import pandas as pd
import numpy as np
from statistics import mean
from analysis_word_vectors import get_word_vector_from_text, get_vector_cosine_similarity
from analysis_stylometry import get_stylometry_dict_from_text
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob

def compare_row_sentence(raisin_vectors_df, sentence_vector, sentence_vector_similarities):
    # Select only the columns with the vectors
    raisin_vectors_df_vectors = raisin_vectors_df.iloc[:, 2:]
    # Convert the input vector to a numpy array and reshape it
    sentence_vector_np = np.array(sentence_vector).reshape(1, -1)
    # Calculate the cosine similarity
    cosine_similarities = cosine_similarity(raisin_vectors_df_vectors, sentence_vector_np)
    # Get the max cosine similarity
    cosine_similarity_max = cosine_similarities.max()
    # Append to sentence vector similarities
    sentence_vector_similarities.append(cosine_similarity_max)

def compare_file(file_path, raisin_vectors_df, raisin_stylometry_df):
    # Read file
    with open(file_path, "r") as f:
        text = f.read()
        # Get sentence blob
        blob = TextBlob(text)
        # Get stylometry dict
        inner_stylometry_dict = get_stylometry_dict_from_text(blob)
        # Substract the values of the stylometry dict from the stylometry df
        external_stylometry_diff_df = raisin_stylometry_df.apply( lambda x: abs(x - inner_stylometry_dict[x.name]) if x.name in inner_stylometry_dict else x)
        external_stylometry_diff = external_stylometry_diff_df.min().to_dict()
        # Prepend stylometry_dict keys with "inner_"
        inner_stylometry_dict = { f"inner_{key}": value for key, value in inner_stylometry_dict.items() }
        # Prepend stylometry_diff keys with "external_"
        external_stylometry_diff = { f"external_{key}": value for key, value in external_stylometry_diff.items() }
        # Get sentence vectors
        sentence_vectors = [ get_word_vector_from_text(sentence) for sentence in blob.sentences ]
        # Store all sentence vector similarities
        sentence_vectors_similarities = []
        # Iterate over all sentence vectors
        for i, sentence_vector in enumerate(sentence_vectors):
            print(f"\t\x1b[2m{file_path} Sentence {i} / {len(sentence_vectors)} processed\x1b[0m")
            compare_row_sentence(raisin_vectors_df, sentence_vector, sentence_vectors_similarities)
        # Get the max sentence similarity
        sentence_vector_similarity_max = max(sentence_vectors_similarities)
        # Get the average sentence similarity
        sentence_vector_similarity_avg = mean(sentence_vectors_similarities) 
        # Return the dict
        return {
            **inner_stylometry_dict,
            **external_stylometry_diff,
            "sentence_vector_similarity_max": sentence_vector_similarity_max,
            "sentence_vector_similarity_avg": sentence_vector_similarity_avg
        }

def main():
    print("HIIII")
    # Get directory path, raisin_vectors path and raisin_stylometry path from command line arguments
    if len(sys.argv) > 3:
        dir_path = sys.argv[1]
        raisin_vectors_path = sys.argv[2]
        raisin_stylometry_path = sys.argv[3]
    else:
        raise Exception("Please provide a directory path, raisin_vectors path and raisin_stylometry path")
    # Get the dataframes
    raisin_vectors_df = pd.read_csv(raisin_vectors_path)
    raisin_stylometry_df = pd.read_csv(raisin_stylometry_path)
    # Analyze file
    print(
        compare_file(dir_path, raisin_vectors_df, raisin_stylometry_df)
    )

if __name__ == "__main__":
    main()