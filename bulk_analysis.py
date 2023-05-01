import sys
import statistics
import pprint
import json
from textblob import TextBlob
from n_grams_analysis import n_grams_analyze
from vector_analysis import vector_analyze, vector_similarity_cosine

def intrinsic_analysis(file_path):
    with open(file_path, "r") as f:
        # Read document
        document = f.read()
        blob = TextBlob(document)
        # Literal n-grams
        n_grams_literal = n_grams_analyze(blob)
        # PoS n-grams
        blob_pos = " ".join([t[1] for t in blob.tags])
        n_grams_pos = n_grams_analyze(TextBlob(blob_pos))
        sentence_lengths_arr = [len(s.words) for s in blob.sentences]
        sentence_length = statistics.stdev(sentence_lengths_arr) if len(sentence_lengths_arr) > 1 else 0

        # Return the results as a dictionary
        return {
            "n_grams_literal_integrity": n_grams_literal,
            "n_grams_pos_integrity": n_grams_pos,
            "sentence_length": sentence_length
        }
    
def bulk_intrinsic_analysis(directory):
    # Store results
    results = []
    # Iterate over all files in directory
    for file_path in os.listdir(directory):
        # Analyze the file
        results.append(intrinsic_analysis(file_path))
    # Store results in a csv file
    with open("results.csv", "w") as f:
        f.write("n_grams_literal_integrity,n_grams_pos_integrity,sentence_length\n")
        for result in results:
            f.write(str(result["n_grams_literal_integrity"]) + "," + str(result["n_grams_pos_integrity"]) + "," + str(result["sentence_length"]) + "\n")

def main():
    try:
        # Read file name from command line arguments
        if len(sys.argv) > 1:
            directory = sys.argv[1]
        else:
            raise Exception("No directory specified")

        print("Analyzing directory " + directory + "...")
        # Analyze the directory
        bulk_intrinsic_analysis(directory)
        # Print the results
        print("Done")
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    main()