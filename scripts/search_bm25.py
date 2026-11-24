"""
BM25 baseline retrieval for ClimateLens.

The query goes through the SAME preprocess_text() function Raina used on
the documents (lowercase -> tokenize -> stopwords -> Porter stem), so
query terms match document terms exactly.

Examples:
  python scripts/search_bm25.py "renewable energy projects in India"
  python scripts/search_bm25.py "flood" --topic extreme_weather -k 5
  python scripts/search_bm25.py "carbon tax" --from 2026-11-01 --sort date
  python scripts/search_bm25.py --test        # run the test queries, save CSV
"""
import argparse
from pathlib import Path

import pandas as pd
from opensearchpy import OpenSearch

from preprocess_text import preprocess_text   # Raina's function, same folder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
INDEX_NAME = "climatelens"

TEST_QUERIES = [
    "renewable energy projects reducing carbon emissions",
    "heatwave and wildfire damage",
    "carbon tax policy",
    "climate adaptation for coastal flooding",
    "drought impact on crops and farmers",
    "electric vehicles and transport emissions",
    "biodiversity loss and ecosystem protection",
    "net zero targets of governments",
]

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}], use_ssl=False)


def build_query(processed_query, k, topic=None, domain=None,
                date_from=None, date_to=None, sort="relevance"):
    filters = []
    if topic:
        filters.append({"term": {"topic": topic}})
    if domain:
        filters.append({"term": {"domain": domain}})
    if date_from or date_to:
        date_range = {}
        if date_from:
            date_range["gte"] = date_from
        if date_to:
            date_range["lte"] = date_to
        filters.append({"range": {"date": date_range}})

    body = {
        "size": k,
        "query": {
            "bool": {
                # "must" is scored with BM25
                "must": [{"match": {"processed_text": processed_query}}],
                # "filter" narrows results but does NOT change scores
                "filter": filters,
            }
        },
        "_source": ["title", "desc", "url", "topic", "domain", "outletName", "date"],
    }
    if sort == "date":
        body["sort"] = [{"date": "desc"}, "_score"]
    return body


def search(query, k=10, **kwargs):
    """Return a list of result dicts. Import this from FastAPI / evaluation code."""
    processed_query = preprocess_text(query)
    if not processed_query:
        return []          # query was only stopwords / symbols

    body = build_query(processed_query, k, **kwargs)
    response = client.search(index=INDEX_NAME, body=body)

    results = []
    for rank, hit in enumerate(response["hits"]["hits"], start=1):
        src = hit["_source"]
        results.append({
            "rank": rank,
            "score": hit["_score"],
            "doc_id": hit["_id"],
            "title": src["title"],
            "desc": src.get("desc"),
            "topic": src["topic"],
            "source": src.get("outletName") or src["domain"],
            "date": src["date"],
            "url": src["url"],
        })
    return results


def print_results(query, results):
    print(f'\nQuery: "{query}"  ->  processed: "{preprocess_text(query)}"')
    if not results:
        print("  No results.")
        return
    for r in results:
        score = f"{r['score']:.3f}" if r["score"] is not None else "-"
        print(f"{r['rank']:>2}. [{score}] {r['title']}")
        print(f"      {r['topic']} | {r['source']} | {r['date'][:10]}")
        print(f"      {r['url']}")


def run_test_queries(k):
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for q in TEST_QUERIES:
        results = search(q, k=k)
        print_results(q, results)
        for r in results:
            rows.append({"query": q, **r})
    out = RESULTS_DIR / "bm25_baseline_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} rows to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="search text")
    parser.add_argument("-k", type=int, default=10, help="number of results")
    parser.add_argument("--topic", help="e.g. renewable_energy, extreme_weather")
    parser.add_argument("--domain", help="e.g. reuters.com")
    parser.add_argument("--from", dest="date_from", help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", help="YYYY-MM-DD")
    parser.add_argument("--sort", choices=["relevance", "date"], default="relevance")
    parser.add_argument("--test", action="store_true", help="run built-in test queries")
    args = parser.parse_args()

    if args.test:
        run_test_queries(args.k)
    elif args.query:
        results = search(args.query, k=args.k, topic=args.topic, domain=args.domain,
                         date_from=args.date_from, date_to=args.date_to, sort=args.sort)
        print_results(args.query, results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
