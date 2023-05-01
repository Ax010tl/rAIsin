import sys
import statistics
import pprint
import json
import os
import pandas as pd
from datetime import datetime
from textblob import TextBlob
from n_grams_analysis import n_grams_analyze
from vector_analysis import vector_analyze, vector_similarity_cosine

def analyze_file_extrinsic(file_path, raisin_vectors_path):
    with open(file_path, "r") as f, open(raisin_vectors_path, "r") as v:
        # Read document
        document = f.read()
        blob = TextBlob(document)
        # Read vectors
        raisin_vectors = json.load(v)
        file_vector = vector_analyze(blob)
        # Iterate over all documents to find most similar
        vector_document_closest = 0
        for document in raisin_vectors:
            # Don't compare to self
            if document["file_name"] == os.path.basename(file_path):
                continue
            vector_document_closest = max( vector_similarity_cosine(file_vector["vector"], document["vector"]), vector_document_closest )
        return {
            "vector_document_closest": vector_document_closest
        }

def analyze_file_intrinsic(file_path):
    with open(file_path, "r") as f:
        document = f.read()
        blob = TextBlob(document)
        # Literal n-grams
        n_grams_literal = n_grams_analyze(blob)
         # PoS n-grams
        blob_pos = " ".join([t[1] for t in blob.tags])
        n_grams_pos = n_grams_analyze(TextBlob(blob_pos))
        sentence_lengths_arr = [len(s.words) for s in blob.sentences]
        sentence_length = statistics.stdev(sentence_lengths_arr) if len(sentence_lengths_arr) > 1 else 0
        # Return
        return {
            "file_path": file_path,
            "n_grams_literal_integrity": n_grams_literal,
            "n_grams_pos_integrity": n_grams_pos,
            "sentence_length": sentence_length,
        }
        

def analyze_file(file_path, raisin_vectors_path):
    intrinsic = None
    intrinsic = analyze_file_intrinsic(file_path)
    extrinsic = analyze_file_extrinsic(file_path, raisin_vectors_path)
    return {**intrinsic, **extrinsic}

def bulk_analysis(directory, raisin_vectors_path):
    # Store results
    results = []
    # Iterate over all files in directory
    for file_path in os.listdir(directory):
        results.append( 
            analyze_file(os.path.join(directory, file_path), raisin_vectors_path) 
        )
    # Store results
    df = pd.DataFrame(results)
    df.to_csv(f"results_{datetime.now().isoformat()}.csv")

def main():
    try:
        # Read dir name from command line arguments
        if len(sys.argv) > 2:
            dir_name = sys.argv[1]
            raisin_vectors_path = sys.argv[2]
        else:
            raise Exception("No raisin_vectors.json file path specified")

        print("Analyzing file " + dir_name + "...")
        # Analyze the file
        results = bulk_analysis(dir_name, raisin_vectors_path)
        # Print the results
        pprint.pprint(results)
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    main()