"""
BM25 written by hand (no OpenSearch) - for understanding and cross-checking.

It builds an inverted index in Python and scores with the same BM25 formula
OpenSearch/Lucene uses. Its top results should closely match search_bm25.py.
(Scores differ slightly because Lucene stores document lengths approximately.)

Run:  python scripts/bm25_from_scratch.py "carbon tax policy"
"""
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from preprocess_text import preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "climate_news_preprocessed.csv"

K1 = 1.2
B = 0.75


def build_inverted_index(docs):
    """docs: list of token lists. Returns postings {term: {doc_idx: tf}} and doc lengths."""
    postings = defaultdict(dict)
    doc_lengths = []
    for doc_idx, tokens in enumerate(docs):
        doc_lengths.append(len(tokens))
        for term, tf in Counter(tokens).items():
            postings[term][doc_idx] = tf
    return postings, doc_lengths


def bm25_search(query_tokens, postings, doc_lengths, k=10):
    N = len(doc_lengths)
    avgdl = sum(doc_lengths) / N
    scores = defaultdict(float)

    for term in set(query_tokens):
        if term not in postings:
            continue
        n = len(postings[term])                              # docs containing term
        idf = math.log(1 + (N - n + 0.5) / (n + 0.5))        # Lucene's IDF
        for doc_idx, tf in postings[term].items():
            dl = doc_lengths[doc_idx]
            norm = K1 * (1 - B + B * dl / avgdl)
            scores[doc_idx] += idf * (tf * (K1 + 1)) / (tf + norm)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def main():
    query = " ".join(sys.argv[1:]) or "renewable energy projects reducing carbon emissions"

    df = pd.read_csv(INPUT_FILE)
    df = df[df["processed_text"].fillna("").str.strip() != ""].reset_index(drop=True)
    docs = df["processed_text"].str.split().tolist()

    postings, doc_lengths = build_inverted_index(docs)
    print(f"Documents: {len(docs)} | Vocabulary size: {len(postings)} | "
          f"Avg doc length: {sum(doc_lengths) / len(doc_lengths):.1f} tokens")

    q_tokens = preprocess_text(query).split()
    print(f'\nQuery: "{query}"  ->  tokens: {q_tokens}')
    for t in q_tokens:
        print(f"  '{t}' appears in {len(postings.get(t, {}))} documents")

    print()
    for rank, (doc_idx, score) in enumerate(bm25_search(q_tokens, postings, doc_lengths), 1):
        row = df.iloc[doc_idx]
        print(f"{rank:>2}. [{score:.3f}] {row['title']}  ({row['topic']})")


if __name__ == "__main__":
    main()
