import sys
from textblob import TextBlob
from n_grams_analysis import n_grams_analyze

def analyze_file(file_path):
    with open(file_path, "r") as f:
        corpus = f.read()
        blob = TextBlob(corpus)
        # N-GRAMS ANALYSIS
        ## Literal
        n_grams_literal = n_grams_analyze(blob)
        ## PoS
        # Get the PoS tags without the original word
        blob_pos = " ".join([t[1] for t in blob.tags])
        n_grams_pos = n_grams_analyze(TextBlob(blob_pos))
        # Return the results as a dictionary
        return {
            "n_grams_literal": n_grams_literal,
            "n_grams_pos": n_grams_pos
        }

def main():
    try:
        # Read file name from command line arguments
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
        else:
            throw("No file path provided")
        print("Analyzing file " + file_path + "...")
        # Analyze the file
        results = analyze_file(file_path)
        # Print the results
        print(results)
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    main()