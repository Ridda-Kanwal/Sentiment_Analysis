import re
import string
from typing import List

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------------------------------
# ONE-TIME SETUP: make sure the required NLTK corpora/models are present.
# Safe to run every time — nltk.download() no-ops (quietly) if already cached.
# --------------------------------------------------------------------------
def ensure_nltk_data():
    required = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]
    for path, pkg_id in required:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg_id, quiet=True)

ensure_nltk_data()

# --------------------------------------------------------------------------
# STEP 1 & 2: TEXT PRE-PROCESSING PIPELINE
# (Tokenization -> Negation-Safe Stop-Word Removal -> POS-Guided Lemmatization)
# --------------------------------------------------------------------------
class TextPreprocessor:
  
    # Words that carry negation / polarity-flipping meaning and must survive
    # stop-word filtration even though NLTK's default list marks them as stop words.
    NEGATION_WORDS = {
        "not", "no", "nor", "none", "never", "neither",
        "n't", "cannot", "can't", "won't", "don't", "doesn't",
        "didn't", "isn't", "aren't", "wasn't", "weren't",
        "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't",
        "couldn't", "without",
    }

    def __init__(self):
        self.stop_words = set(stopwords.words("english")) - self.NEGATION_WORDS
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _character_normalize(text: str) -> str:
        """Strip HTML remnants (e.g. <br>) and collapse repeated punctuation/case noise."""
        text = re.sub(r"<[^>]+>", " ", text)          # <br>, <p>, etc.
        text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
        text = re.sub(r"[^a-zA-Z'\s]", " ", text)       # keep letters + apostrophes
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _treebank_to_wordnet(tag: str) -> str:
        """Map a Penn Treebank POS tag to the tag WordNetLemmatizer expects."""
        if tag.startswith("J"):
            return "a"   # adjective
        if tag.startswith("V"):
            return "v"   # verb
        if tag.startswith("N"):
            return "n"   # noun
        if tag.startswith("R"):
            return "r"   # adverb
        return "n"       # default assumption

    def clean(self, text: str) -> List[str]:
        # -- Character normalization + lowercasing (funnel diagram) --
        text = self._character_normalize(text).lower()

        # -- Tokenization --
        tokens = word_tokenize(text)

        # -- Negation-safe stop-word filtration --
        tokens = [t for t in tokens if t not in self.stop_words and t not in string.punctuation]

        # -- POS-guided lemmatization ("Context is Everything") --
        tagged = nltk.pos_tag(tokens)
        lemmas = [
            self.lemmatizer.lemmatize(word, self._treebank_to_wordnet(tag))
            for word, tag in tagged
        ]
        return lemmas

    def transform(self, documents: List[str]) -> List[str]:
        """Return cleaned documents as space-joined strings, ready for TfidfVectorizer."""
        return [" ".join(self.clean(doc)) for doc in documents]


# --------------------------------------------------------------------------
# STEP 3 & 4: VECTORIZATION (TF-IDF, n-grams, bounded dimensionality, CSR)
# --------------------------------------------------------------------------
def build_vectorizer(max_features: int = 5000, min_df: int = 2) -> TfidfVectorizer:
 
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
    )


# --------------------------------------------------------------------------
# STEP 5: MODEL TRAINING (Naive Bayes with Laplace smoothing, or SVM)
# --------------------------------------------------------------------------
def train_model(X_train: csr_matrix, y_train, model_type: str = "naive_bayes",
                 imbalanced: bool = False, alpha: float = 1.0):
  
    if model_type == "svm":
        model = LinearSVC()
    elif imbalanced:
        model = ComplementNB(alpha=alpha)
    else:
        model = MultinomialNB(alpha=alpha)

    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------
# END-TO-END PIPELINE
# --------------------------------------------------------------------------
class SentimentPipeline:
    def __init__(self, model_type: str = "naive_bayes", imbalanced: bool = False,
                 max_features: int = 5000, min_df: int = 2, alpha: float = 1.0):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = build_vectorizer(max_features=max_features, min_df=min_df)
        self.model_type = model_type
        self.imbalanced = imbalanced
        self.alpha = alpha
        self.model = None

    def fit(self, X_train_raw: List[str], y_train):
        cleaned = self.preprocessor.transform(X_train_raw)
        # Ensure the matrix is explicitly stored as CSR sparse (Memory Optimization step)
        X_vec = csr_matrix(self.vectorizer.fit_transform(cleaned))
        self.model = train_model(
            X_vec, y_train,
            model_type=self.model_type,
            imbalanced=self.imbalanced,
            alpha=self.alpha,
        )
        return self

    def predict(self, X_raw: List[str]):
        cleaned = self.preprocessor.transform(X_raw)
        X_vec = csr_matrix(self.vectorizer.transform(cleaned))
        return self.model.predict(X_vec)

    def evaluate(self, X_raw: List[str], y_true):
        y_pred = self.predict(X_raw)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        return acc, report, cm

# --------------------------------------------------------------------------
# DEMO: SAMPLE PRODUCT-REVIEW DATASET
# (No dataset was attached to the brief, so a small illustrative one is
#  included here — swap in your own CSV of reviews/labels via load_csv().)
# --------------------------------------------------------------------------
def sample_dataset() -> pd.DataFrame:
    reviews = [
        ("This product is TERRIBLE!!! <br> Waste of money.", "negative"),
        ("Absolutely wonderful, I am not disappointed at all.", "positive"),
        ("The battery life is not good, died in an hour.", "negative"),
        ("Great quality and fast shipping, highly recommend!", "positive"),
        ("I am not happy with this purchase, it broke immediately.", "negative"),
        ("Works exactly as described, very satisfied.", "positive"),
        ("Not worth the price, cheaply made and flimsy.", "negative"),
        ("Excellent build quality, exceeded my expectations.", "positive"),
        ("This is not the best purchase but not the worst either.", "negative"),
        ("Superb customer service and a fantastic product overall.", "positive"),
        ("Do not buy this, it stopped working after two days.", "negative"),
        ("I love how well this was packaged and it works perfectly.", "positive"),
        ("The screen is not bright enough and the app crashes.", "negative"),
        ("Solid value for the money, would buy again.", "positive"),
        ("Never received the item, terrible experience overall.", "negative"),
        ("A truly delightful product, my whole family enjoys it.", "positive"),
        ("Not impressed, the material feels cheap and thin.", "negative"),
        ("Best purchase I've made this year, works flawlessly.", "positive"),
        ("Horrible experience, customer support never responded.", "negative"),
        ("Highly durable and comfortable, no complaints at all.", "positive"),
    ]
    return pd.DataFrame(reviews, columns=["review", "sentiment"])

def load_csv(path: str, text_col: str = "review", label_col: str = "sentiment") -> pd.DataFrame:
    """Load a real dataset: expects at least a text column and a label column."""
    df = pd.read_csv(path)
    return df[[text_col, label_col]].rename(columns={text_col: "review", label_col: "sentiment"})

def main():
    df = sample_dataset()
    print(f"Loaded {len(df)} labeled reviews.")
    print("(This is a tiny illustrative dataset — swap in load_csv('your_reviews.csv') "
          "with a few thousand rows for realistic accuracy.)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"].tolist(), df["sentiment"].tolist(),
        test_size=0.3, random_state=42, stratify=df["sentiment"]
    )

    pipeline = SentimentPipeline(model_type="naive_bayes", imbalanced=False, alpha=1.0)
    pipeline.fit(X_train, y_train)

    acc, report, cm = pipeline.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2%}\n")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Sanity check on the exact negation example from the 'Stop-Word Trap' slide
    print("\n--- Negation sanity check ---")
    for text in ["I am not happy.", "I am happy.", "This is not good at all."]:
        pred = pipeline.predict([text])[0]
        print(f"{text!r:35s} -> {pred}")

if __name__ == "__main__":
    main()
