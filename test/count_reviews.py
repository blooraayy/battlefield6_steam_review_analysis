import pandas as pd

df = pd.read_csv("data/pragmata_reviews_raw.csv")
print(df["language"].value_counts())