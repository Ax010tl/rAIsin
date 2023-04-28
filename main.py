from textblob import TextBlob

def analyze_n_grams(corpus):
    blob = TextBlob(corpus)
    n_grams = blob.ngrams(n=3)
    return n_grams

def main():
    corpus = "This is a corpus of text to be analyzed."
    n_grams = analyze_n_grams(corpus)
    print(n_grams)

if __name__ == "__main__":
    main()