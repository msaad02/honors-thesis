"""
This script is pretty simple. We are taking in the evaluation data that we
created in the `1_eval_create.py` script and then evaluating it on each of
the models that we have created in this project, that is, the scratch model,
finetuned model, and retrieval augmented generation (RAG).

These results will be saved and then evaluated by GPT-4 later.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # prevent tensorflow logs

# Set path to parent directory so we can import from other folders.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fine_tuning.finetune_class import FineTunedModel
from scratch_model.inference import ScratchModel
from text_search.rag import RAG
from termcolor import colored
from tqdm import tqdm
import pandas as pd

# ----- Loading in models/data -----
models = {
    "RAG": RAG(
        main_categorization_model_dir="../text_search/models/main_category",
        subcategorization_model_dir="../text_search/models/subcategory_models/",
        embeddings_file="../text_search/data/embeddings.pickle"),
    "Finetuned": FineTunedModel(repo_id="msaad02/BrockportGPT-7b"),
    "Scratch": ScratchModel(model_dir="../scratch_model/models/transformer_v7/")
}

data = pd.read_csv('./data/evaluation_data.csv').sample(n=10)
questions = data["question"].to_list()


# ----- Generic eval function -----
def eval_model(name: str, model, **kwargs) -> list:
    "Evaluate the model for all the questions in the test data."

    print(colored(f"Running {name} model...", "blue"))

    responses = []
    for question in tqdm(questions):
        try:
            responses.append(model(question, **kwargs))
        except:
            responses.append("Error")

    return responses


# ----- Running the models -----
for name, model in models.items():
    args = {}
    
    if name == "RAG":
        args['config'] = {
            "search_type": "typesense",
            "use_classifier": False,
            "n_results": 5,
            "model_kwargs": {"temperature": 0.0},
        }

    data[name] = eval_model(name, model, **args)


# ----- Saving the results -----
data.to_csv("./data/evaluation_output.csv", index=False)
