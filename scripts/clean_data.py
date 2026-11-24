import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "renewable_energy_gdelt.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "renewable_energy_clean.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper functions

def clean_text(text):
    """
    Basic whitespace/text cleanup.
    Does NOT stem, tokenize, or remove stopwords yet.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Collapse repeated whitespace/newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_url(url):
    """
    Normalize URLs for duplicate detection.
    Removes fragments, tracking parameters, and :443.
    """
    if pd.isna(url):
        return ""

    url = str(url).strip()

    # Fix examples like example.com:443/article
    url = url.replace(":443/", "/")

    try:
        parts = urlsplit(url)

        # lowercase hostname
        netloc = parts.netloc.lower()

        # Remove common tracking parameters
        tracking_params = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "fbclid",
            "gclid"
        }

        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in tracking_params
        ]

        normalized = urlunsplit(
            (
                parts.scheme.lower(),
                netloc,
                parts.path.rstrip("/"),
                urlencode(query),
                ""  # remove fragment
            )
        )

        return normalized

    except Exception:
        return url


# Load raw data


df = pd.read_csv(RAW_FILE)

print("RAW DATA")
print("-" * 40)
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()


# 1. Remove rows missing essential fields

df = df.dropna(subset=["title", "url"])


# 2. Keep English articles

df["lang"] = df["lang"].astype(str).str.lower().str.strip()
df = df[df["lang"] == "en"]


# 3. Clean text fields

text_columns = [
    "domain",
    "outletName",
    "title",
    "desc",
    "author"
]

for column in text_columns:
    df[column] = df[column].apply(clean_text)



# 4. Convert date to proper datetime

df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

# Remove rows whose dates could not be parsed
df = df[df["date"].notna()]


# 5. Normalize URLs


df["normalized_url"] = df["url"].apply(normalize_url)


# 6. Remove duplicate URLs

before = len(df)

df = df.drop_duplicates(
    subset=["normalized_url"],
    keep="first"
)

print(f"Duplicate URLs removed: {before - len(df)}")


# 7. Remove exact duplicate titles

# Lowercase version only for duplicate detection
df["normalized_title"] = (
    df["title"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

before = len(df)

df = df.drop_duplicates(
    subset=["normalized_title"],
    keep="first"
)

print(f"Duplicate titles removed: {before - len(df)}")


# 8. Create a searchable text field


# For now, combine title and description.
# Full article text can be added later if your team crawls URLs.
df["search_text"] = (
    df["title"].fillna("")
    + " "
    + df["desc"].fillna("")
).str.strip()


# 9. Flag missing descriptions

df["desc_missing"] = df["desc"].eq("")


# 10. Sort by publication date

df = df.sort_values(
    by="date",
    ascending=False
).reset_index(drop=True)


# 11. Save clean dataset


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("CLEANED DATA")
print("-" * 40)
print(f"Rows remaining: {len(df)}")
print(f"Missing descriptions: {df['desc_missing'].sum()}")
print(f"Output saved to: {OUTPUT_FILE}")