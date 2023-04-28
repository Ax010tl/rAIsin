from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
import numpy as np
from numpy.linalg import norm
from textblob import TextBlob
import os
import sys
import json

model = api.load("glove-wiki-gigaword-50")# train a model from the corpus

def vector_from_text(text_blob):
    vectors = np.zeros(50)
    for word in text_blob.words:
        try:
            vectors+=model.get_vector(word)
            # print(word)
        except:
            pass
    # Return the average vector
    return vectors / len(text_blob.words)


def vector_similarity_cosine(vec1, vec2):
    return np.dot(vec1, vec2)/(norm(vec1)*norm(vec2))

def vector_analyze(document_blob):
    result = {
        "vector": None, 
        "sentences": []
    }
    # Get the vector of the document
    result["vector"] = vector_from_text(document_blob).tolist()
    # Iterate over document paragraphs
    for sentence in document_blob.sentences:
        # Get sentence blob
        sentence_blob = TextBlob(sentence.raw)
        # Get the vector of the sentence
        sentence_vector = vector_from_text(sentence_blob)
        # Append
        result["sentences"].append({
            "vector": sentence_vector.tolist(),
        })

    # Return the result
    return result

def vector_train_directory(directory_path):
    # Store
    documents = []
    # Get all files in the directory
    files = os.listdir(directory_path)
    # Iterate over the files
    for file_name in files:
        # Get the file path
        file_path = os.path.join(directory_path, file_name)
        # Read the file
        with open(file_path, "r") as f:
            document = f.read()
            blob = TextBlob(document)
            # Analyze the document
            result = vector_analyze(blob)
            # Store the result
            result["file_name"] = file_name
            documents.append(result)
    # Store as a json file inside the directory
    with open(os.path.join(directory_path, "raisin_vectors.json"), "w") as f:
        f.write(json.dumps(documents))
    # Return the documents
    return documents


def main():
    try:
        # Read file name from command line arguments
        if len(sys.argv) > 1:
            directory_path = sys.argv[1]
        else:
            throw("No directory path provided")
        print("Analyzing directory " + directory_path + "...")
        # Train the directory
        results = vector_train_directory(directory_path)
        # Print the results
        print(results)
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    main()