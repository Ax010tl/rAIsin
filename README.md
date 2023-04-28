# rAIsin
Plagiarism detection using n-grams and word2vec

# Usage
```
python3 main.py name_of_file
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
