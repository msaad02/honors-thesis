"""
This script is pretty simple. We are taking in the test data that we
split in the `data_collection/5_upload_datasets.py` script and then
evaluating it on all the models that we have created in this project.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # prevent tensorflow logs

# Set path to parent directory so we can import from other folders.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fine_tuning.finetune_class import FineTunedModel
from scratch_model.inference import ScratchModel
from text_search.rag import RAG
from torch.cuda import is_available
from termcolor import colored
from datasets import load_dataset
from tqdm import tqdm

# ----- Loading in models/data -----
models = {
    "RAG": RAG(
        main_categorization_model_dir="../text_search/models/main_category_model",
        subcategorization_model_dir="../text_search/models/subcategory_models/",
        embeddings_file="../text_search/data/embeddings.pickle"),
    "Finetuned": FineTunedModel(model_type="gptq" if is_available() else "gguf"),
    "Scratch": ScratchModel(model_dir="../scratch_model/models/transformer_v7/")
}

data = load_dataset("msaad02/brockport-gpt-4-qa")['test'].to_pandas()
data = data.loc[:1, ["question", "answer"]]
questions = data["question"].to_list()


# ----- Generic eval function -----
def eval_model(name: str, model, **kwargs) -> list:
    "Evaluate the model for all the questions in the test data."

    print(colored(f"Running {name} model...", "blue"))

    responses = []
    for question in tqdm(questions):
        responses.append(model(question, **kwargs))

    return responses


# ----- Running the models -----
for name, model in models.items():
    args = {}
    
    if name == "RAG":
        args['config'] = {
            "search_type": "typesense",
            "use_classifier": True,
            "n_results": 3,
            "model_kwargs": {"temperature": 0.8},
        }

    data[name] = eval_model(name, model, **args)


# ----- Saving the results -----
data.to_csv("answers.csv", index=False)