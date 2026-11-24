import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "renewable_energy_clean.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "renewable_energy_relevant.csv"

df = pd.read_csv(INPUT_FILE)

RENEWABLE_TERMS = [
    "renewable energy",
    "solar power",
    "solar energy",
    "solar farm",
    "solar project",
    "wind power",
    "wind energy",
    "wind farm",
    "clean energy",
    "green energy",
    "energy storage",
    "battery storage",
    "hydrogen production",
    "green hydrogen",
    "renewable electricity",
    "renewable power",
    "energy transition"
]

def relevance_score(row):
    title = str(row.get("title", "")).lower()
    desc = str(row.get("desc", "")).lower()

    title_matches = sum(term in title for term in RENEWABLE_TERMS)
    desc_matches = sum(term in desc for term in RENEWABLE_TERMS)

    # Give title matches more weight
    return (title_matches * 2) + desc_matches


df["relevance_score"] = df.apply(relevance_score, axis=1)

# Keep articles with at least one reasonably strong renewable-energy signal
df["renewable_relevant"] = df["relevance_score"] >= 2

relevant_df = df[df["renewable_relevant"]].copy()

relevant_df.to_csv(OUTPUT_FILE, index=False)

print(f"Input rows: {len(df)}")
print(f"Relevant rows: {len(relevant_df)}")
print(f"Removed as weak/off-topic: {len(df) - len(relevant_df)}")
print(f"Saved to: {OUTPUT_FILE}")