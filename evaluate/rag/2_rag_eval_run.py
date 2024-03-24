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
import pandas as pd
from tqdm import tqdm
import ast

tqdm.pandas()

df = pd.read_csv("../data/rag_evaluation_data.csv").loc[:10,:]

model = RAG(
    main_categorization_model_dir="../../text_search/models/main_category",
    subcategorization_model_dir="../../text_search/models/subcategory_models/",
    embeddings_file="../../text_search/data/embeddings.pickle"
)

df['response'] = df.progress_apply(lambda x: model(x["question"], ast.literal_eval(x["player"]), verbose=True), axis=1)

df.to_csv("../data/rag_evaluation_output.csv", index=False)