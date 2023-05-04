import pandas as pd
import os
import sys
from compare_file import compare_file

def analyze_directory(directory, raisin_vectors_df, raisin_stylometry_df):
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
        result = compare_file(file_path, raisin_vectors_df, raisin_stylometry_df)
        # Store result
        results.append(result)
        print(f"\x1b[2m{i} - File {file_name} processed\x1b[0m")
    # Store as a csv file
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(directory, "raisin_comparison_no_vector.csv"))
    # Return results
    return results

def main():
    print("HIIIII")
    # Get directory path, raisin vectors and stylometry data from command line arguments
    if len(sys.argv) > 3:
        directory = sys.argv[1]
        raisin_vectors_path = sys.argv[2]
        raisin_stylometry_path = sys.argv[3]
    else:
        raise Exception("Please provide directory path, raisin vectors path and raisin stylometry path as command line arguments")
    # Read raisin vectors and stylometry data
    raisin_vectors_df = pd.read_csv(raisin_vectors_path)
    raisin_stylometry_df = pd.read_csv(raisin_stylometry_path)
    # Analyze directory
    analyze_directory(directory, raisin_vectors_df, raisin_stylometry_df)
    # Print success message
    print("Success")

if __name__ == "__main__":
    main()