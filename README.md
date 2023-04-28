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