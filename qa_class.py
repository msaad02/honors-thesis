"""
This is the main class for using everything created in this project.
It basically just manages the question answering process, since each
subsection of this project also has its own class for interfacing with
the user. Mainly, this class is used by the Gradio UI to get the answer
to a question. But it also is nice for users generally.

Please note that this is only designed for STREAMING responses currently.

To use it, just create an instance of this class and then call the
`run_model` method with the question and config options. The config
options are a dictionary with the following keys:

    model_type: str (finetuned, scratch, or rag)
    search_type: str (semantic, semantic_rerank, typesense, or none)
    use_classifier: bool (True or False)
    n_results: int (Number of search results to return)
    model_kwargs: dict (Arguments for the model such as temperature)

Example usage:

```
from question_answering import Manage_QA

# Create the question client
question_client = Manage_QA()

# Ask a question
question = "How can I apply to Brockport?"

# Define the model configuration
config = {
    "model_type": "rag", # ["rag", "finetuned", "scratch"]
    "search_type": "typesense", # ["typesense", "semantic_rerank", "semantic", "none"], 
    "use_classifier": True, # [True, False]
    "n_results": 3, # [1, 2, 3, 4, 5, ...]
    "model_kwargs": {"temperature": 0.8}, # see openai or ctransformers docs for more info
}

# You can optionally do stuff with the generator, but this just prints the results
for result in question_client.run_model(question, config):
    pass

# Note there is a way to reduce the amount of printing, but this is nice for understanding
print('\n'+'-'*150)
print(result)
```

Which will print out a the config, question, context retrieves, and answer.
You can reduce the amount of printing but adjusting "category_logging" and 
"verbose" in RAG. Other models do not print anything other than the answer.
"""


from fine_tuning.finetune_class import FineTunedModel
from scratch_model.inference import ScratchModel
from text_search.rag import RAG
from torch.cuda import is_available
from termcolor import colored
from typing import Optional

class Manage_QA:
    def __init__(
        self,
        finetuned_model_name: str = "msaad02/BrockportGPT-7b",
        scratch_model_dir: str = "./scratch_model/models/transformer_v7/",
        main_categorization_model_dir: str = "./text_search/models/main_category",
        subcategorization_model_dir: str = "./text_search/models/subcategory_models/",
        embeddings_file: str = "./text_search/data/embeddings.pickle",
        typesense_host: str = "localhost",
        typesense_port: str = "8108",
        typesense_protocol: str = "http",
        typesense_collection_name: str = "brockport_data_v1",
        typesense_api_key: str = "xyz",
        device: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        print(colored("Loading Models...\n", color="red", attrs=["bold"]))
        if device is None:
            device = "cuda" if is_available() else "cpu"

        self.rag = RAG(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
            embeddings_file=embeddings_file,
            typesense_host=typesense_host,
            typesense_port=typesense_port,
            typesense_protocol=typesense_protocol,
            typesense_collection_name=typesense_collection_name,
            typesense_api_key=typesense_api_key,
            device=device,
            openai_api_key=openai_api_key,
            category_logging=True,  # Print categories to console or not - I find this helpful
        )

        self.finetuned_model = FineTunedModel(
            repo_id=finetuned_model_name,
            stream=True
        )

        self.scratch_model = ScratchModel(model_dir=scratch_model_dir)

        print(colored("Models Loaded!\n", color="green", attrs=["bold"]))

    def run_model(self, question, config):
        "Gets the answer to the question using the specified model type."

        model_type = config["model_type"]
        model_kwargs = config["model_kwargs"]

        assert model_type in ["finetuned", "scratch", "rag"]

        # Trying to get the answer from the model chosen
        try:
            if model_type == "finetuned":
                # Finetuned uses the model_kwargs
                response = self.finetuned_model(question, **model_kwargs)

            elif model_type == "scratch":
                # Does not use any of the other config options
                response = self.scratch_model(question, stream=True)

            elif model_type == "rag":
                # RAG requires the other config options unlike the others
                response = self.rag(question, config, stream=True, verbose=True)
            
            for result in response:
                yield result

        except:
            yield "Error generating response."


    def __del__(self):
        "Clean up when the program is interrupted."

        del self.rag
        del self.finetuned_model
        del self.scratch_model