import sys
import statistics
import pprint
from textblob import TextBlob
from n_grams_analysis import n_grams_analyze

def analyze_file(file_path):
    with open(file_path, "r") as f:
        document = f.read()
        blob = TextBlob(document)
        ## Literal n-grams
        n_grams_literal = n_grams_analyze(blob)
        ## PoS n-grams
        blob_pos = " ".join([t[1] for t in blob.tags])
        n_grams_pos = n_grams_analyze(TextBlob(blob_pos))
        # TODO: Get standard deviation of sentence length
        sentence_lengths_arr = [len(s.words) for s in blob.sentences]
        sentence_length = statistics.stdev(sentence_lengths_arr) if len(sentence_lengths_arr) > 1 else 0

        # Return the results as a dictionary
        return {
            "n_grams_literal_integrity": n_grams_literal,
            "n_grams_pos_integrity": n_grams_pos,
            "sentence_length": sentence_length
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
        pprint.pprint(results)
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    main()