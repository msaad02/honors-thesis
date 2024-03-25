"""
This script creates the data for RAG evaluation.

NOTE: This relies on the `evaluate/1_eval_create.py` script to be run first.
"""

import pandas as pd
from itertools import product
import random

qas = pd.read_csv("../data/evaluation_data.csv")

# Configuration options
choices = {
    "search_type": ["typesense", "semantic_rerank", "semantic"],
    "use_classifier": [True, False],
    "n_results": [1, 2, 3, 4, 5],
    "temperature": [0, 0.5, 0.8, 0.1]
}

# Generate all combinations of configurations
all_combinations = list(product(*choices.values()))

# Create a DataFrame from the combinations, lots of stuff happens...
df = pd.DataFrame(all_combinations, columns=choices.keys())
df["model_kwargs"] = df.apply(lambda x: {"temperature": x["temperature"]}, axis=1)
df = df.drop(columns=["temperature"])
df = pd.DataFrame({"player": [{**row[1]} for row in df.iterrows()]})

# qas = qas.groupby("type").sample(n=len(qas))
# qas = qas.sample(frac=1).reset_index(drop=True)

df = df.sample(n=len(qas), replace=True).reset_index(drop=True)
df = df.rename({"player": "player_A_config"}, axis=1)
df['player_A_config'] = df['player_A_config'].apply(str)

def pick_opponent(player: str):
    players = set(df['player_A_config'])
    players.remove(player)
    return random.choice(list(players))
    
df['player_B_config'] = df['player_A_config'].apply(pick_opponent)

df_out = pd.concat([qas, df], axis=1)
df_out.to_csv("../data/rag_evaluation_data.csv", index=False)
