import pandas as pd
import os
import sys

INDEX_COL = "IME I PREZIME"
TOP_N = int(sys.argv[1])

results = {file: pd.read_csv(file).dropna(subset=[INDEX_COL], how="all") for file in os.listdir() if file.startswith("kolo") and file.endswith(".csv")}
columns = results["kolo1.csv"].columns
numerical_columns = [col for col in columns if results["kolo1.csv"][col].dtype.kind in "fi"]

scaled_results = []
for name, r in results.items():
    assert all(r.columns == columns)
    if r[INDEX_COL].shape[0] != r[INDEX_COL].nunique():
        print("duplicate players")
        exit()
    
    r[INDEX_COL] = r[INDEX_COL].str.strip()
    r[INDEX_COL] = r[INDEX_COL].str.replace("-", " ")
    r.drop("TOTAL", axis=1, inplace=True)
    r["TOTAL"] = r.sum(axis=1)

    scaled_r = r.copy(deep=True)
    scaled_r[numerical_columns] /= scaled_r[numerical_columns].max()
    scaled_r[numerical_columns] *= 100
    scaled_results.append(scaled_r)

def user_collect(data):
    return data.sort_values(ascending=False).head(TOP_N).sum()

total = pd.concat(scaled_results, ignore_index=True)
total_grouped = total.groupby(INDEX_COL).agg(user_collect)
total_grouped.drop("GRAD", axis=1, inplace=True)

for col in total_grouped:
    print(total_grouped[col].sort_values(ascending=False).head())
    print()