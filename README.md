# rAIsin
Plagiarism detection using n-grams and word2vec

# Usage
```
python3 main.py name_of_file raisin_vectors_file
```

# N-Grams
N-grams are a contiguous sequence of n items from a given sample of text. We are using n-grams to detect variations of style within a given text. We analyze literal n-gram frequencies and PoS n-gram frequencies.
The returned result shows the average n-gram frequency of the text. The lower the frequency, the more differences identified within the text.

See also: https://ceur-ws.org/Vol-502/paper8.pdf

# Word Vectors
Word vectors are a numerical representation of a word's meaning. We are using word vectors to detect variations of meaning within a given text. We use existing GloVe word vectors to analyze the text and return the average cosine similarity between words in the text. The lower the cosine similarity, the more differences identified within the text.

See also:
1. https://nlp.stanford.edu/pubs/glove.pdf
2. https://github.com/RaRe-Technologies/gensim-data
3. https://medium.com/@akankshagupta371/understanding-text-summarization-using-k-means-clustering-6487d5d37255

In order to make comparisons easier and more efficient, vectors generated for each document and its sentences are stored in a `raisin_vectors.json` file. To generate this file, place all the target documents in a directory. Then call
```
python3 vector_analysis.py name_of_directory
```

<details><summary>Dataset</summary>


Corpus is a shortened version of [PAN-PC-11](https://webis.de/data/pan-pc-11.html), selecting texts between 4500-5500 words. They are divided in the following:

[clean](./data/clean/) --> 740 texts without any plagiarism

[plag](./data/plag/) --> 706 texts with different types of plagiarism (translation, obfuscated, among others)

[sus](./data/sus/) --> 40 texts, some with plagiarism, some without (meant to help us determine our model's efficiency)

Each text has an accompanying `.xml`, with data regarding plagiarism (if any) and source.

</details>