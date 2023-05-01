import sys
import statistics
import pprint
import json
import os
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
            vector_document_closest = max( vector_similarity_cosine(file_vector["vector"], document["vector"]), vector_document_closest )
        

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
    return {
        "extrinsic": extrinsic, 
        "intrinsic": intrinsic
    }

def bulk_intrinsic_analysis(directory):
    # Store results
    results = []
    # Iterate over all files in directory
    for file_path in os.listdir(directory):
        results.append( analyze_file_intrinsic(os.path.join(directory, file_path)) )
    # Store results
    with open(f"intrinsic_results_{datetime.now().isoformat()}.csv", "w") as f:
        f.write("file_path,n_grams_literal_integrity,n_grams_pos_integrity,sentence_length\n")
        for result in results:
            f.write(result["file_path"] + "," + str(result["n_grams_literal_integrity"]) + "," + str(result["n_grams_pos_integrity"]) + "," + str(result["sentence_length"]) + "\n")

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
        results = bulk_intrinsic_analysis(dir_name)
        # Print the results
        pprint.pprint(results)
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    main()