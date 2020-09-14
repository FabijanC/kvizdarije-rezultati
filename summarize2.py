import pandas as pd
import os
import sys
from pathlib import Path
import numpy as np

INDEX_COL = "IME I PREZIME"
AGG_DIR = sys.argv[1]
TOP_N = int(sys.argv[2])
PLAYED_SO_FAR = 8

NUMERIC_COLS = ["Kvizdarije " + str(i) for i in range(1, PLAYED_SO_FAR+1)]
USECOLS = NUMERIC_COLS + [INDEX_COL]
CONVERTERS = {
    col: lambda val: 0.0 if val.strip() == "-" else float(val) for col in NUMERIC_COLS
}
CONVERTERS[INDEX_COL] = lambda val: val.strip()

topic2res = {
    topic.strip(".csv").upper(): pd.read_csv(Path(AGG_DIR, topic), converters=CONVERTERS, usecols=USECOLS, index_col=INDEX_COL) for topic in os.listdir(AGG_DIR)
}

for round_i in range(9, 10+1):
    round = pd.read_csv(f"kolo{round_i}.csv", index_col=INDEX_COL, converters={INDEX_COL: lambda name: name.strip()})
    round = round.loc[round.index.dropna()]
    round.drop("GRAD", axis=1, inplace=True)

    assert set(topic2res.keys()) == set(round.columns)

    #normalize
    round = round / round.max() * 100

    for topic in topic2res:
        res = topic2res[topic]
        res[f"Kvizdarije {round_i}"] = round[topic]

for topic in topic2res:
    res = topic2res[topic]
    res[res.isna()] = 0.0
    res["total"] = np.sum(np.partition(topic2res[topic].values, -TOP_N)[:,-TOP_N:], 1)

    print(topic)
    print(res.sort_values("total", ascending=False).head())
    print()