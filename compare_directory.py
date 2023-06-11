import pandas as pd
import os
import sys
from compare_file import compare_file
import argparse

def analyze_directory(directory, raisin_vectors_df, raisin_stylometry_df, raisin_word_frequency_df):
    # Store results
    results = []
    # Iterate over all files
    for i, file_name in enumerate(os.listdir(directory)):
        # Ignore non-txt files
        if not file_name.endswith(".txt"):
            continue
        # Get file path
        file_path = os.path.join(directory, file_name)
        # Analyze file
        result = compare_file(file_path, raisin_vectors_df, raisin_stylometry_df, raisin_word_frequency_df)
        # Store result
        results.append(result)
        print(f"\x1b[2m{i} - File {file_name} processed\x1b[0m")
    # Store as a csv file
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(directory, "raisin_comparison_no_vector.csv"))
    # Return results
    return results

if __name__ == "__main__":
    # Get the arguments
    parser = argparse.ArgumentParser(description="Analyze a directory of text files. Produces a csv file with the results of the analysis")
    parser.add_argument("directory", help="Directory path")
    parser.add_argument("raisin_vectors_path", help="Path to the csv file containing the raisin sentence vectors")
    parser.add_argument("raisin_stylometry_path", help="Path to the csv file containing the raisin stylometry data")
    parser.add_argument("raisin_word_frequency_path", help="Path to the csv file containing the raisin word frequency data")
    args = parser.parse_args()
    # Read csv files
    raisin_vectors_df = pd.read_csv(args.raisin_vectors_path)
    raisin_stylometry_df = pd.read_csv(args.raisin_stylometry_path)
    raisin_word_frequency_df = pd.read_csv(args.raisin_word_frequency_path)
    # Analyze directory
    analyze_directory(args.directory, raisin_vectors_df, raisin_stylometry_df, raisin_word_frequency_df)
    # Print success message
    print("Success")