"""
With the evaluation data, now we can run the RAG model for each
configuration and question in the evaluation data CSV file.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # prevent tensorflow logs

# # Set path to parent directory so we can import from other folders.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from text_search.rag import RAG
import concurrent.futures
import pandas as pd
from tqdm import tqdm
import time
import ast

tqdm.pandas()

df = pd.read_csv("../data/rag_eval_v1/rag_evaluation_data.csv")
# df = df.sample(n=3)

model = RAG(
    main_categorization_model_dir="../../text_search/models/main_category",
    subcategorization_model_dir="../../text_search/models/subcategory_models/",
    embeddings_file="../../text_search/data/embeddings.pickle"
)

# ------------- Parallelzing the evaluation ------------------------------------------------------

def process_item(row, t: tqdm, player: str = "A"):
    "Process a single item in the evaluation data."

    start_time = time.time()
    
    try:
        if player == "A":
            response = model(row[1]['question'], ast.literal_eval(row[1]['player_A_config']))
        else:
            response = model(row[1]['question'], ast.literal_eval(row[1]['player_B_config']))
    except:
        response = None

    tqdm.write(f"Success! Complete in {time.time() - start_time:.2f}s")
    tqdm.update(t)

    return row[0], response

gpt_output = {}
t = tqdm(total=len(df)*2)

for player in ["A", "B"]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_item = {executor.submit(process_item, row, t=t, player=player): row[0] for row in df.iterrows()}
        for future in concurrent.futures.as_completed(future_to_item):
            key = future_to_item[future]
            try:
                idx, response = future.result()
                gpt_output[idx] = response
            except Exception as exc:
                print('%r generated an exception: %s' % (idx, exc))

    if player == "A":
        df["player_A_response"] = df.index.map(gpt_output)
    else:
        df["player_B_response"] = df.index.map(gpt_output)


t.close()

df.to_csv("../data/rag_evaluation_output.csv", index=False)

print("\nDone!\n", flush=True)