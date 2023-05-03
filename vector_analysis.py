from gensim.models.word2vec import Word2Vec
from textblob import Word
import gensim.downloader as api
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
from numpy.linalg import norm
from textblob import TextBlob
import os
import sys
import json

# Use the pre-trained model for word2vec
model = api.load("glove-wiki-gigaword-100")

"""
Get the vector of a text blob
The vector for each lemmatized word is calculated and then averaged
"""
def vector_from_text(text_blob):
    # Store the vector of n dimensions
    vectors = np.zeros(100)
    for word in text_blob.words:
        try:
            # Lemmatize the word
            w = Word(word).lemma
            # Check if stopword or not in model
            if w in STOPWORDS or w not in model:
                continue
            # Add the vector
            vectors += model.get_vector(w)
        except:
            pass
    # Return the average vector
    return vectors / len(text_blob.words)

"""
The cosine similarity between two vectors measures the similarity between 
two non-zero vectors defined in an inner product space.
It is obtained by dividing the dot product of the two vectors by the product
"""
def vector_similarity_cosine(vec1, vec2):
    return np.dot(vec1, vec2)/(norm(vec1)*norm(vec2))

"""
Analyze a document and return the vector of the document and the vectors of
each sentence
"""
def vector_analyze(document_blob):
    # Store the result
    result = {
        "vector": None, 
        "sentences": []
    }
    # Get the vector of the document
    result["vector"] = vector_from_text(document_blob).tolist()
    # Iterate over document paragraphs
    """for sentence in document_blob.sentences:
        # Get sentence blob
        sentence_blob = TextBlob(sentence.raw)
        # Get the vector of the sentence
        sentence_vector = vector_from_text(sentence_blob)
        # Append
        result["sentences"].append({
            "vector": sentence_vector.tolist(),
        })
    """
    # Return the result
    return result

"""
Analyze a directory of documents and store the result as a json file
"""
def vector_train_directory(directory_path):
    # Store all the documents
    documents = []
    # Iterate over all the files in a directory
    for file_name in os.listdir(directory_path):
        # Ignore non-text files
        if not file_name.endswith(".txt"):
            continue
        # Get the complete file path
        file_path = os.path.join(directory_path, file_name)
        # Read the file
        with open(file_path, "r") as f:
            # Get the document blob
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

"""
This is the main function that is called when the script is run
It reads the directory path from the command line arguments and calls the
vector_train_directory function to store the results in a json file
"""
def main():
    try:
        # Read file name from command line arguments
        if len(sys.argv) > 1:
            directory_path = sys.argv[1]
        else:
            raise Exception("No directory path provided")
        print("Analyzing directory " + directory_path + "...")
        # Train the directory
        results = vector_train_directory(directory_path)
        # Print the results
        print(results)
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    main()