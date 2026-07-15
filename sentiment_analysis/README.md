# Sentiment Analysis Pipeline

An end-to-end NLP pipeline that classifies product reviews as **Positive** or **Negative**, built from scratch using NLTK and scikit-learn. Rather than relying on a single black-box function, this project implements every stage of the text-to-prediction process explicitly — preprocessing, feature engineering, and classification.

## Features

- **Negation-aware stop-word removal** — filters common low-signal words (*the, is, at*) while explicitly preserving negations (*not, never, isn't*), so phrases like "not happy" aren't misread as positive.
- **POS-guided lemmatization** — tags each token's part of speech before lemmatizing, so words reduce correctly (e.g., *"went" → "go"*) instead of defaulting to noun-only reduction.
- **TF-IDF vectorization with n-grams** — captures both single words and two-word phrases (e.g., "not good"), with bounded vocabulary size to keep memory usage manageable.
- **Sparse matrix storage** — all feature matrices are stored in SciPy's CSR format for efficiency.
- **Configurable classifier** — Multinomial or Complement Naive Bayes (with Laplace smoothing) or Linear SVM.
- **Zero manual setup** — required NLTK corpora (tokenizer, stopwords, WordNet, POS tagger) download automatically on first run.

## Tech Stack

Python 3 · NLTK · scikit-learn · SciPy · pandas

## Installation

```bash
git clone https://github.com/Ridda-Kanwal/Sentiment_Analysis.git
cd Sentiment_Analysis
pip install nltk scikit-learn scipy pandas
```

## Usage

Run with the built-in demo dataset:

```bash
python sentiment_analysis.py
```

To train on your own data, use the CSV loader:

```python
from sentiment_analysis import load_csv, SentimentPipeline
from sklearn.model_selection import train_test_split

df = load_csv("your_reviews.csv", text_col="review", label_col="sentiment")
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.3, random_state=42
)

pipeline = SentimentPipeline(model_type="naive_bayes")
pipeline.fit(X_train, y_train)

acc, report, cm = pipeline.evaluate(X_test, y_test)
print(report)
```

## How It Works

1. **Clean** — strip HTML/URLs, lowercase, tokenize.
2. **Filter stop-words** — remove low-signal words, keep negations.
3. **Lemmatize** — POS-tag first, then reduce each word to its base form.
4. **Vectorize** — convert text to TF-IDF vectors (unigrams + bigrams).
5. **Classify** — train Naive Bayes (or SVM) and predict sentiment.

## Notes

The bundled dataset (20 reviews) is illustrative only. For meaningful accuracy, use a real dataset of at least a few thousand labeled reviews (e.g., Amazon or Yelp review datasets).

## License

MIT
