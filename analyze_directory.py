from textblob import TextBlob
import pandas as pd
import os
import sys
from analysis_word_vectors import get_word_vector_from_text
from analysis_stylometry import get_stylometry_dict_from_text
from analysis_frequency import get_word_frequency_matrix
from sklearn.feature_extraction.text import CountVectorizer
import argparse

# Get sentence vectors from files in a directory
def get_sentence_vectors_from_directory(dir_path):
    # Store all rows
    rows = []
    # Iterate over all files
    for file_name in os.listdir(dir_path):
        # Ignore non-txt files
        if not file_name.endswith(".txt"):
            continue
        # Get file path
        file_path = os.path.join(dir_path, file_name)
        # Read file
        with open(file_path, "r") as f:
            text = f.read()
            # Get sentence blob
            blob = TextBlob(text)
            # Iterate over all sentences
            for sentence in blob.sentences:
                # Get vector
                vector = get_word_vector_from_text(sentence)
                # Store file name
                vector.insert(0, file_name)
                # Store row
                rows.append(vector)
            print(f"\x1b[2mFile {file_name} processed\x1b[0m")
    # Store as a csv file
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(dir_path, "raisin_vectors.csv"))

# Get stylometric features from files in a directory
def get_stylometry_from_directory(dir_path):
    # Store all rows
    rows = []
    # Iterate over all files
    for file_name in os.listdir(dir_path):
        # Ignore non-txt files
        if not file_name.endswith(".txt"):
            continue
        # Get file path
        file_path = os.path.join(dir_path, file_name)
        # Read file
        with open(file_path, "r") as f:
            text = f.read()
            blob = TextBlob(text)
            # Get stylometry dict
            stylometry_dict = get_stylometry_dict_from_text(blob)
            # Store file name
            stylometry_dict["file_name"] = file_name
            # Store row
            rows.append(stylometry_dict)
            print(f"\x1b[2mFile {file_name} processed\x1b[0m")
    # Store as a csv file
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(dir_path, "raisin_stylometry.csv"))

def get_word_frequency_from_directory(dir_path):
    # Store all documents
    documents = []
    # Iterate over all files
    for file_name in os.listdir(dir_path):
        # Ignore non-txt files
        if not file_name.endswith(".txt"):
            continue
        # Get file path
        file_path = os.path.join(dir_path, file_name)
        # Read file
        with open(file_path, "r") as f:
            text = f.read()
            documents.append(text)
            print(f"\x1b[2mFile {file_name} processed\x1b[0m")
    # Get word frequency matrix
    df = get_word_frequency_matrix(documents)
    # Store as a csv file
    df.to_csv(os.path.join(dir_path, "raisin_word_frequency.csv"))


# Get all features from files in a directory
def get_all_features_from_directory(dir_path):
    get_sentence_vectors_from_directory(dir_path)
    get_stylometry_from_directory(dir_path)
    get_word_frequency_from_directory(dir_path)

def main():
    # Read directory path from command line arguments
    if len(sys.argv) > 1:
        dir_path = sys.argv[1]
    else:
        raise Exception("Directory path not provided")
    # Get sentence vectors from files in a directory
    get_all_features_from_directory(dir_path)
    # Print success message
    print("Success")

if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser(description="Analyze a directory of text files")
    parser.add_argument("dir_path", help="The path to the directory")
    parser.add_argument('analysis_type', help='The type of analysis to perform', choices=['glove_vectors', 'stylometry', 'word_frequency', 'all'], default='all')
    args = parser.parse_args()
    # If the directory path is not provided
    if not args.dir_path:
        raise Exception("Directory path not provided")
    # Switch between analysis types
    if args.analysis_type == "glove_vectors":
        get_sentence_vectors_from_directory(args.dir_path)
    elif args.analysis_type == "stylometry":
        get_stylometry_from_directory(args.dir_path)
    elif args.analysis_type == "word_frequency":
        get_word_frequency_from_directory(args.dir_path)
    elif args.analysis_type == "all":
        get_all_features_from_directory(args.dir_path)
    # Print success message
    print("Success")