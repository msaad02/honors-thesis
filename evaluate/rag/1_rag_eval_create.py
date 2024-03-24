"""
This script creates the data for RAG evaluation.
"""

import pandas as pd
from itertools import product

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
df = df.sample(n=len(qas), replace=True).reset_index(drop=True)

# Now we have a length ~2300 of configurations, we want to combine it with the questions
out = pd.concat([qas, df], axis=1)

out.to_csv("../data/rag_evaluation_data.csv", index=False)


