"""
Build the ClimateLens OpenSearch index from Raina's preprocessed CSV.

Input : data/processed/climate_news_preprocessed.csv
Output: OpenSearch index "climatelens" on http://localhost:9200

Key idea: the text is ALREADY stemmed in Python (processed_text column),
so OpenSearch must NOT stem it again. We index processed_text with the
"whitespace" analyzer, which only splits on spaces.

Run:  python scripts/build_index.py
      python scripts/build_index.py --k1 1.5 --b 0.6   (try other BM25 settings)
"""
import argparse
from pathlib import Path

import pandas as pd
from opensearchpy import OpenSearch, helpers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "climate_news_preprocessed.csv"
INDEX_NAME = "climatelens"


def get_client():
    return OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)


def index_definition(k1, b):
    return {
        "settings": {
            "number_of_shards": 1,      # 1 shard -> exact corpus-wide BM25 statistics
            "number_of_replicas": 0,
            "similarity": {
                "climate_bm25": {"type": "BM25", "k1": k1, "b": b}
            },
        },
        "mappings": {
            "properties": {
                # Searched field: stemmed tokens, scored with BM25
                "processed_text": {
                    "type": "text",
                    "analyzer": "whitespace",
                    "similarity": "climate_bm25",
                },
                # Display fields (shown to the user, not used for ranking)
                "title": {"type": "text"},
                "desc": {"type": "text"},
                "url": {"type": "keyword", "index": False},
                "author": {"type": "keyword"},
                # Filter / sort fields: exact values
                "topic": {"type": "keyword"},
                "domain": {"type": "keyword"},
                "outletName": {"type": "keyword"},
                "date": {"type": "date"},
                "relevance_score": {"type": "integer"},
            }
        },
    }


def load_documents():
    df = pd.read_csv(INPUT_FILE)

    # Drop rows with nothing to search on
    df = df[df["processed_text"].fillna("").str.strip() != ""]

    # Dates look like "2026-11-24 02:30:03+00:00" -> convert to ISO "2026-11-24T02:30:03Z"
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Replace NaN with None so it becomes JSON null
    df = df.astype(object).where(pd.notna(df), None)
    return df


def generate_actions(df):
    for _, row in df.iterrows():
        yield {
            "_index": INDEX_NAME,
            "_id": row["normalized_url"],          # unique ID -> no duplicates
            "_source": {
                "processed_text": row["processed_text"],
                "title": row["title"],
                "desc": row["desc"],
                "url": row["url"],
                "author": row["author"],
                "topic": row["topic"],
                "domain": row["domain"],
                "outletName": row["outletName"],
                "date": row["date"],
                "relevance_score": row["relevance_score"],
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    client = get_client()
    print("Connected to OpenSearch", client.info()["version"]["number"])

    # Start fresh every time (BM25 settings can only be set at creation)
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        print(f"Deleted old index '{INDEX_NAME}'")

    client.indices.create(index=INDEX_NAME, body=index_definition(args.k1, args.b))
    print(f"Created index '{INDEX_NAME}' with BM25 k1={args.k1}, b={args.b}")

    df = load_documents()
    print(f"Loaded {len(df)} documents from {INPUT_FILE.name}")

    ok, errors = helpers.bulk(client, generate_actions(df), raise_on_error=False)
    print(f"Indexed {ok} documents, {len(errors)} errors")
    for e in errors[:5]:
        print("  error:", e)

    client.indices.refresh(index=INDEX_NAME)   # make documents searchable now
    count = client.count(index=INDEX_NAME)["count"]
    print(f"Documents in index: {count}")


if __name__ == "__main__":
    main()
