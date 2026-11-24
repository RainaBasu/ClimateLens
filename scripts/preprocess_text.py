import pandas as pd
import re
from pathlib import Path
from nltk.stem import PorterStemmer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "climate_news_all.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "climate_news_preprocessed.csv"
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "for", "from", "had", "has", "have", "he", "her",
    "hers", "him", "his", "i", "if", "in", "into", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "ours", "she",
    "so", "that", "the", "their", "theirs", "them", "they", "this",
    "those", "to", "too", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "will", "with", "you", "your",
    "yours", "about", "after", "again", "against", "all", "am",
    "any", "because", "before", "between", "both", "can", "did",
    "do", "does", "doing", "down", "during", "each", "few", "further",
    "here", "how", "more", "most", "no", "nor", "not", "now",
    "off", "once", "only", "other", "out", "over", "own", "same",
    "should", "some", "such", "than", "then", "there", "these",
    "through", "under", "until", "up", "very", "while", "why"
}

STEMMER = PorterStemmer()


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", " ", text)

    tokens = re.findall(r"[a-z0-9]+", text)

    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
        and len(token) > 1
    ]

    tokens = [
        STEMMER.stem(token)
        for token in tokens
    ]

    return " ".join(tokens)


def main():
    df = pd.read_csv(INPUT_FILE)

    print("PREPROCESSING CLIMATELENS CORPUS")
    print(f"Input rows: {len(df)}")

    df["search_text"] = (
        df["title"].fillna("")
        + " "
        + df["desc"].fillna("")
    ).str.strip()

    df["processed_text"] = df["search_text"].apply(preprocess_text)

    empty_processed = (
        df["processed_text"]
        .str.strip()
        .eq("")
        .sum()
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Processed rows: {len(df)}")
    print(f"Rows with empty processed text: {empty_processed}")
    print(f"Saved to: {OUTPUT_FILE}")

    print()
    print("EXAMPLE")

    sample = df.iloc[0]

    print()
    print("Title:")
    print(sample["title"])

    print()
    print("Original search text:")
    print(sample["search_text"])

    print()
    print("Processed text:")
    print(sample["processed_text"])


if __name__ == "__main__":
    main()