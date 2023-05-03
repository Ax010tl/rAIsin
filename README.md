# rAIsin
Plagiarism detection using n-grams and word2vec

# Usage
```
python3 main.py name_of_file raisin_vectors_file
```

# Processing
Before training, processing is done to obtain variables that are then fed into the model. This involve metrics for both intrinsic and extrinsic analysis.

## N-Grams
N-grams are a contiguous sequence of n items from a given sample of the text. They can be used to detect variations of style within a given text. We analyze literal n-gram frequencies and PoS n-gram frequencies.
Using the formula from [this paper](https://ceur-ws.org/Vol-502/paper8.pdf), we obtain the average n-gram frequency of the text. The lower the frequency, the more differences identified within the text.
$$
nd_1(A, B) = \frac{ \sum_{g\in P(A)} \left( \frac{2(f_A(g) -f_B(g))}{f_A(g) + f_B(g)} \right)^2 }{ 4|P(A)| }
$$

This process is done for both literal and PoS n-grams.

## Word Vectors
Word vectors are a numerical representation of a word's meaning. To get a vector for a document, we average the vectors of all the words in the document. We then compare the vectors of the documents to each other using cosine similarity. The lower the cosine similarity, the more differences identified within the text.

In order to make comparisons easier and more efficient, vectors generated for each document and its sentences are stored in a `raisin_vectors.json` file. To generate this file, place all the target documents in a directory. Then call
```
python3 vector_analysis.py name_of_directory
```

### Results
Resullts are stored in csv files with the columns:


<details><summary>Dataset</summary>


Corpus is a shortened version of [PAN-PC-11](https://webis.de/data/pan-pc-11.html), selecting texts between 4500-5500 words. They are divided in the following:

[clean](./data/clean/) --> 740 texts without any plagiarism

[plag](./data/plag/) --> 706 texts with different types of plagiarism (translation, obfuscated, among others)

[sus](./data/sus/) --> 40 texts, some with plagiarism, some without (meant to help us determine our model's efficiency)

Each text has an accompanying `.xml`, with data regarding plagiarism (if any) and source.

</details>