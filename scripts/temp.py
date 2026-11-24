#To manually inspect certain entries

import pandas as pd

relevant = pd.read_csv("data/processed/renewable_energy_relevant.csv")
clean = pd.read_csv("data/processed/renewable_energy_clean.csv")

print(
    relevant[["title", "desc", "relevance_score"]]
    .sample(30, random_state=42)
    .to_string(index=False)
)