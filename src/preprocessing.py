import pandas as pd
path = "data/processed/BunnyGirl300.csv"

df = pd.read_csv(path)

df = df.dropna()


# Add a sentence length column
df["length"] = df["text"].str.len()