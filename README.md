# rAIsin
Plagiarism detection using n-grams and word2vec

# Usage

## Analyzing existing data
To analyze existing data and save the results as a `raisin_vectors.csv`, `raisin_stylometry.csv` and `raisin_word_frequency.csv` files, place all the target documents in a directory. Then call
```
python3 analyze_directory.py name_of_directory [action]
```
Where action can be `glove_vectors`, `stylometry`, `word_frequency` or `all`. If no action is specified, `all` is used.

The resulting files will be used when comparing new documents to the existing data.

## Comparing a file
To compare a file with the results in `raisin_vectors.csv` and `raisin_stylometry.csv`, call
```
python3 compare_file.py name_of_file path_to_raisin_vectors path_to_raisin_stylometry
```

## Comparing a directory
To compare a directory with the results in `raisin_vectors.csv` and `raisin_stylometry.csv`, call
```
python3 compare_directory.py name_of_directory path_to_raisin_vectors path_to_raisin_stylometry
```

This is used to compare a directory of files to the existing data. The results are stored in a `raisin_results.csv` file. These file is used to train the model.

## Training the model
This program receives two `raisin_results.csv` files, one with the results of the clean texts and one with the results of the plagiarized texts. It then trains a model using these results.
```
python3 train_model.py path_to_clean_raisin_results path_to_plag_raisin_results
```

The model is saved as `raisin_model.h5`.

## Making predictions
Finally, the model can be used to predict if a text is plagiarized or not. To do this, call
```
python3 predict.py path_to_file path_to_model path_to_raisin_vectors path_to_raisin_stylometry
```

# Analysis
Data is analyzed from each document into quantitative variables. These can be divided into two categories: intrinsic and extrinsic.

## Intrinsic
Intrinsic variables are those that are obtained from the text itself. Stylometry is a branch of linguistics that studies the characteristics of a text. Variations in these characteristics can be used to identify sections of a text that are not written by the same author. When analyzing a directory, results are saved in the `raisin_stylometry.csv` file. The following variables are used for intrinsic analysis:

### N-Grams
N-grams are a contiguous sequence of n items from a given sample of the text. They can be used to detect variations of style within a given text. We analyze literal n-gram frequencies and PoS n-gram frequencies.
Using the formula from [this paper](https://ceur-ws.org/Vol-502/paper8.pdf), we obtain the average n-gram frequency of the text. The lower the frequency, the more differences identified within the text.
$$
nd_1(A, B) = \frac{ \sum_{g\in P(A)} \left( \frac{2(f_A(g) -f_B(g))}{f_A(g) + f_B(g)} \right)^2 }{ 4|P(A)| }
$$

### Sentence length
Sentence length is also a stylistic feature that can be used to identify differences in a text. Both the _average_ and _normalized standard deviation_ of the sentence length are used. The higher the standard deviation, the more differences identified within the text.

### Word length
Word length is also a stylistic feature that can be used to identify differences in a text. Both the _average_ and _normalized standard deviation_ of the word length are used. The higher the standard deviation, the more differences identified within the text.

### Syllables per word
Syllables per word is also a stylistic feature that can be used to identify differences in a text. (*) Both the _average_ and _normalized standard deviation_ of the syllables per word are used. The higher the standard deviation, the more differences identified within the text.
> (*) This feature might not be as relevant as the others, considering it is not strictly a stylistic feature, but rather a feature of the language itself. Especially in scientific texts.

### Flesh-Kincaid Reading Ease Score
The [Flesch-Kincaid Reading Ease Score](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests) is a measure of how easy it is to read a text. The higher the score, the easier it is to read. The lower the score, the more difficult it is to read. This is a stylistic feature that can be used to identify differences in a text.

This feature is evaluated at the sentence level, and both the _average_ and _normalized standard deviation_ are used. The higher the standard deviation, the more differences identified within the text.

### Gunning Fog Index
The [Gunning Fog Index](https://en.wikipedia.org/wiki/Gunning_fog_index) is another measure of how easy it is to read a text. It is also a stylistic feature that can be used to identify differences in a text. 

This feature is evaluated at the sentence level, and both the _average_ and _normalized standard deviation_ are used. The higher the standard deviation, the more differences identified within the text.


## Extrinsic
Extrinsic variables are those that are obtained by comparing the text to other texts. When analyzing a directory, the results are stored in a `raisin_vectors.csv` file. The following variables are used for extrinsic analysis:

### Vector similarity
Word vectors are a numerical representation of a word's meaning. Sentences can be represented as the average of the word vectors of the words in the sentence. The vector for each sentence when analyzing a directory is saved in the `raisin_vectors.csv` file.

When performing comparisons, each sentence in the analyzed file is compared to each sentence in the `raisin_vectors.csv` file. The highest values per sentence are used. Then, both the maximum and the average of these values are used.

See also:
1. https://nlp.stanford.edu/pubs/glove.pdf
2. https://github.com/RaRe-Technologies/gensim-data
3. https://medium.com/@akankshagupta371/understanding-text-summarization-using-k-means-clustering-6487d5d37255

<details><summary>Dataset</summary>


Corpus is a shortened version of [PAN-PC-11](https://webis.de/data/pan-pc-11.html), selecting texts between 4500-5500 words. They are divided in the following:

[clean](./data/clean/) --> 740 texts without any plagiarism

[plag](./data/plag/) --> 706 texts with different types of plagiarism (translation, obfuscated, among others)

[sus](./data/sus/) --> 40 texts, some with plagiarism, some without (meant to help us determine our model's efficiency)

Each text has an accompanying `.xml`, with data regarding plagiarism (if any) and source.

</details>