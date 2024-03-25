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

df = pd.read_csv("../data/rag_evaluation_data.csv")

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
        elif player == "B":
            response = model(row[1]['question'], ast.literal_eval(row[1]['player_B_config']))
        else:
            raise ValueError("Invalid player")
    except:
        response = None

    tqdm.write(f"Success! Complete in {time.time() - start_time:.2f}s")
    tqdm.update(t)

    return row[0], response


# Assuming df is your DataFrame and process_item is defined

batch_size = 25
t = tqdm(total=len(df)*2)

# Iterate over the DataFrame in batches
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i + batch_size].copy()  # Work with a copy of the current batch

    # Initialize a temporary dictionary for storing responses of the current batch
    gpt_output_a = {}
    gpt_output_b = {}

    for player in ["A", "B"]:
        # Reset the gpt_output dictionary for each player
        gpt_output = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
            future_to_item = {executor.submit(process_item, row, t, player): row[0] for row in batch.iterrows()}
            for future in concurrent.futures.as_completed(future_to_item):
                idx = future_to_item[future]
                try:
                    _, response = future.result()
                    gpt_output[idx] = response
                except Exception as exc:
                    print(f'{idx} generated an exception: {exc}')
        
        # Map the responses to the batch DataFrame
        if player == "A":
            batch["player_A_response"] = batch.index.map(gpt_output)
            gpt_output_a.update(gpt_output)  # Update the batch-specific output dictionary
        else:
            batch["player_B_response"] = batch.index.map(gpt_output)
            gpt_output_b.update(gpt_output)  # Update the batch-specific output dictionary

    # Update the main DataFrame with the processed batch responses
    df.loc[batch.index, "player_A_response"] = batch["player_A_response"]
    df.loc[batch.index, "player_B_response"] = batch["player_B_response"]


t.close()

df.to_csv("../data/rag_evaluation_output.csv", index=False)

print("\nDone!\n", flush=True)