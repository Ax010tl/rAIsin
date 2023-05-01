import sys
import statistics
import pprint
import json
from textblob import TextBlob
from n_grams_analysis import n_grams_analyze
from vector_analysis import vector_analyze, vector_similarity_cosine

def analyze_file(file_path, raisin_vectors_path):
    with open(file_path, "r") as f, open(raisin_vectors_path) as r:
        # Read documents
        document = f.read()
        raisin_vectors_document = json.load(r)
        blob = TextBlob(document)
        # Literal n-grams
        n_grams_literal = n_grams_analyze(blob)
        # PoS n-grams
        blob_pos = " ".join([t[1] for t in blob.tags])
        n_grams_pos = n_grams_analyze(TextBlob(blob_pos))
        sentence_lengths_arr = [len(s.words) for s in blob.sentences]
        sentence_length = statistics.stdev(sentence_lengths_arr) if len(sentence_lengths_arr) > 1 else 0
        # Vector
        file_vector = vector_analyze(blob)
        # VeIterate over all documents to find most similar
        vector_document_closest = float('inf')
        vector_document_closest_name = None
        for document in raisin_vectors_document:
            vector_document_closest = min( vector_similarity_cosine(file_vector["vector"], document["vector"]), vector_document_closest )

        # TODO: Compare with rest of documents
        # Return the results as a dictionary
        return {
            "n_grams_literal_integrity": n_grams_literal,
            "n_grams_pos_integrity": n_grams_pos,
            "sentence_length": sentence_length,
            "vector_document_closest": vector_document_closest
        }

def main():
    try:
        # Read file name from command line arguments
        if len(sys.argv) > 2:
            file_path = sys.argv[1]
            raisin_vectors_path = sys.argv[2]
        else:
            throw("No raisin_vectors.json file path specified")

        print("Analyzing file " + file_path + "...")
        # Analyze the file
        results = analyze_file(file_path, raisin_vectors_path)
        # Print the results
        pprint.pprint(results)
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    main()