import pandas as pd
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "renewable_energy": "renewable_energy_gdelt.csv",
    "extreme_weather": "gdelt_extreme_weather.csv",
    "carbon_emissions": "gdelt_carbon_emission.csv",
    "climate_policy": "gdelt_climate_policy.csv",
    "climate_adaptation": "gdelt_climate_adaptation.csv",
    "agriculture_food": "gdelt_agriculture.csv",
    "transportation": "gdelt_transportation.csv",
    "biodiversity_ecosystem": "gdelt_ecosystem.csv",
}

TOPIC_TERMS = {
    "renewable_energy": [
        "renewable energy",
        "renewable power",
        "renewable electricity",
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
        "green hydrogen",
        "energy transition",
    ],

    "extreme_weather": [
        "extreme weather",
        "heatwave",
        "heat wave",
        "wildfire",
        "drought",
        "flooding",
        "flood",
        "hurricane",
        "cyclone",
        "typhoon",
        "extreme heat",
        "record heat",
        "climate disaster",
        "weather disaster",
    ],

    "carbon_emissions": [
        "carbon emissions",
        "carbon emission",
        "co2 emissions",
        "carbon dioxide emissions",
        "greenhouse gas",
        "greenhouse gases",
        "methane emissions",
        "decarbonization",
        "decarbonisation",
        "carbon footprint",
        "carbon pollution",
        "emissions reduction",
        "emission reduction",
        "net zero emissions",
    ],

    "climate_policy": [
        "climate policy",
        "climate policies",
        "climate law",
        "climate legislation",
        "paris agreement",
        "carbon tax",
        "carbon pricing",
        "net zero",
        "climate target",
        "climate targets",
        "climate regulation",
        "climate regulations",
        "climate agreement",
        "climate summit",
        "cop30",
        "cop31",
        "nationally determined contribution",
        "ndc",
    ],

    "climate_adaptation": [
        "climate adaptation",
        "climate resilience",
        "climate resilient",
        "resilient infrastructure",
        "flood resilience",
        "coastal protection",
        "sea level rise",
        "climate risk",
        "climate risks",
        "adaptation funding",
        "adaptation finance",
        "disaster resilience",
        "climate preparedness",
        "climate vulnerability",
    ],

    "agriculture_food": [
        "climate-smart agriculture",
        "climate smart agriculture",
        "climate agriculture",
        "climate farming",
        "food security",
        "crop loss",
        "crop losses",
        "crop failure",
        "agricultural drought",
        "drought crops",
        "climate resilient crops",
        "climate-resilient crops",
        "sustainable agriculture",
        "sustainable farming",
        "agricultural emissions",
        "farming emissions",
        "food system emissions",
    ],

    "transportation": [
        "transport emissions",
        "transportation emissions",
        "vehicle emissions",
        "electric vehicle",
        "electric vehicles",
        "zero emission vehicle",
        "zero-emission vehicle",
        "clean transportation",
        "clean transport",
        "sustainable transportation",
        "sustainable transport",
        "low carbon transport",
        "low-carbon transport",
        "aviation emissions",
        "shipping emissions",
        "public transit emissions",
        "transport decarbonization",
        "transport decarbonisation",
    ],

    "biodiversity_ecosystem": [
        "biodiversity loss",
        "biodiversity crisis",
        "climate biodiversity",
        "ecosystem loss",
        "ecosystem degradation",
        "habitat loss",
        "deforestation",
        "species loss",
        "species extinction",
        "forest loss",
        "nature loss",
        "ecosystem restoration",
        "biodiversity conservation",
        "climate ecosystem",
    ],
}

CLIMATE_CONTEXT_TERMS = [
    "climate",
    "emission",
    "emissions",
    "carbon",
    "decarbon",
    "renewable",
    "clean energy",
    "green energy",
    "global warming",
    "net zero",
    "sustainable",
    "sustainability",
]

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
}


def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_url(url):
    if pd.isna(url):
        return ""

    url = str(url).strip()
    url = url.replace(":443/", "/")

    try:
        parts = urlsplit(url)

        netloc = parts.netloc.lower()

        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key.lower() not in TRACKING_PARAMS
        ]

        return urlunsplit(
            (
                parts.scheme.lower(),
                netloc,
                parts.path.rstrip("/"),
                urlencode(query),
                "",
            )
        )

    except Exception:
        return url


def normalize_title(title):
    title = clean_text(title).lower()

    title = re.sub(
        r"\s*[-|–—]\s*(reuters|bbc|cnn|forbes|bloomberg|ap news|associated press)$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    return title.strip()


def relevance_score(row, topic):
    title = str(row["title"]).lower()
    desc = str(row["desc"]).lower()

    score = 0

    for term in TOPIC_TERMS[topic]:
        if term in title:
            score += 2

        if term in desc:
            score += 1

    if topic in {
        "agriculture_food",
        "transportation",
        "extreme_weather",
        "biodiversity_ecosystem",
    }:
        full_text = title + " " + desc

        climate_context = any(
            term in full_text
            for term in CLIMATE_CONTEXT_TERMS
        )

        if not climate_context:
            score -= 1

    return score


def process_topic(topic, filename):
    input_file = RAW_DIR / filename

    print()
    print(f"Processing: {topic}")
    print(f"Input: {input_file}")

    df = pd.read_csv(input_file)

    raw_count = len(df)

    df = df.dropna(subset=["title", "url"])

    df["lang"] = (
        df["lang"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = df[df["lang"] == "en"].copy()

    for column in ["domain", "outletName", "title", "desc", "author"]:
        if column in df.columns:
            df[column] = df[column].apply(clean_text)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True
    )

    df = df[df["date"].notna()].copy()

    df["normalized_url"] = df["url"].apply(normalize_url)

    before_urls = len(df)

    df = df.drop_duplicates(
        subset=["normalized_url"],
        keep="first"
    )

    duplicate_urls = before_urls - len(df)

    df["normalized_title"] = df["title"].apply(normalize_title)

    before_titles = len(df)

    df = df.drop_duplicates(
        subset=["normalized_title"],
        keep="first"
    )

    duplicate_titles = before_titles - len(df)

    df["search_text"] = (
        df["title"].fillna("")
        + " "
        + df["desc"].fillna("")
    ).str.strip()

    df["desc_missing"] = df["desc"].eq("")

    df["topic"] = topic

    df["relevance_score"] = df.apply(
        lambda row: relevance_score(row, topic),
        axis=1
    )

    df["topic_relevant"] = df["relevance_score"] >= 2

    cleaned_file = (
        PROCESSED_DIR /
        f"{topic}_clean.csv"
    )

    relevant_file = (
        PROCESSED_DIR /
        f"{topic}_relevant.csv"
    )

    rejected_file = (
        PROCESSED_DIR /
        f"{topic}_rejected.csv"
    )

    df.to_csv(
        cleaned_file,
        index=False
    )

    relevant_df = df[df["topic_relevant"]].copy()

    rejected_df = df[~df["topic_relevant"]].copy()

    relevant_df.to_csv(
        relevant_file,
        index=False
    )

    rejected_df.to_csv(
        rejected_file,
        index=False
    )

    print(f"Raw rows: {raw_count}")
    print(f"Duplicate URLs removed: {duplicate_urls}")
    print(f"Duplicate titles removed: {duplicate_titles}")
    print(f"Clean rows: {len(df)}")
    print(f"Relevant rows: {len(relevant_df)}")
    print(f"Rejected rows: {len(rejected_df)}")

    return relevant_df


def main():
    all_relevant = []

    for topic, filename in FILES.items():
        relevant_df = process_topic(
            topic,
            filename
        )

        all_relevant.append(relevant_df)

    combined = pd.concat(
        all_relevant,
        ignore_index=True
    )

    before_duplicates = len(combined)

    combined = combined.sort_values(
        by="date",
        ascending=False
    )

    combined = combined.drop_duplicates(
        subset=["normalized_url"],
        keep="first"
    )

    combined_duplicate_urls = (
        before_duplicates - len(combined)
    )

    combined = combined.reset_index(drop=True)

    combined_file = (
        PROCESSED_DIR /
        "climate_news_all.csv"
    )

    combined.to_csv(
        combined_file,
        index=False
    )

    print()
    print("Finished processing all topics")
    print(f"Rows before cross-topic deduplication: {before_duplicates}")
    print(f"Cross-topic duplicate URLs removed: {combined_duplicate_urls}")
    print(f"Final corpus rows: {len(combined)}")
    print(f"Saved to: {combined_file}")

    print()
    print("Final topic counts:")
    print(combined["topic"].value_counts())


if __name__ == "__main__":
    main()