from text_search.text_retriever_class import TextRetriever, TypesenseRetriever
import gradio as gr
from openai import OpenAI
from termcolor import colored
from torch.cuda import is_available
from typing import Optional
from ctransformers import AutoModelForCausalLM
from scratch_model.scratch_model_class import ScratchModelEngine
from fine_tuning.finetune_class import FineTunedEngine
import tensorflow as tf

# NOTE TO SELF:

#* Currently openai works with both retrivers.
#* Custom model DOES NOT WORK. Async error for streaming somewhere.
#* Semantic retrieval NOT IMPLEMENTED.
#* Typesesse retrieval is USING CLASSIFIER regardless of what is passed in.
#* Scratch model not in UI, or implemented at all (it also won't stream, yet).

custom_model_prompt = lambda question: f"""\
Below is an inquiry related to SUNY Brockport - from academics, admissions, and faculty support to student life. Prioritize accuracy and brevity.

### Instruction:
{question}

### Response:
"""

openai_system = """You are a helpful chatbot for SUNY Brockport who answers questions using the context given. Be enthusiastic, straightforward, and brief in your responses. Do not answer questions unrelated to SUNY Brockport. If the answer is not clear from the context, say "I'm sorry, I don't know"."""

openai_prompt = lambda context, question: f"Context: {context}\n\nQuestion: {question}"

class Manage_QA:
    def __init__(
        self,
        finetuned_model_name: str = "msaad02/llama2_7b_brockportgpt",
        main_categorization_model_dir: str = "./text_search/models/main_category_model",
        subcategorization_model_dir: str = "./text_search/models/subcategory_models/",
        embeddings_file: str = "./text_search/data/embeddings.pickle",
        typesense_host: str = "localhost",
        typesense_port: str = "8108",
        typesense_protocol: str = "http",
        typesense_collection_name: str = "brockport_data_v1",
        typesense_api_key: str = "xyz",
        use_classifier: bool = True,
        device: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        print(colored("Loading Models...\n", color="red", attrs=["bold"]))
        if device is None:
            device = "cuda" if is_available() else "cpu"
            model_type = "gptq" if device == "cuda" else "gguf"

        self.text_retriever = TextRetriever(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
            embeddings_file=embeddings_file,
            question_classifier=use_classifier,
            device=device
        )

        self.typesense_retriever = TypesenseRetriever(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
            typesense_host=typesense_host,
            typesense_port=typesense_port,
            typesense_protocol=typesense_protocol,
            typesense_collection_name=typesense_collection_name,
            typesense_api_key=typesense_api_key,
            question_classifier=use_classifier
        )

        # self.semantic_retriever = SemanticRetriever()

        self.openai_client = OpenAI(api_key=openai_api_key)

        self.finetuned_model = FineTunedEngine(
            model_name=finetuned_model_name,
            model_type=model_type
        )

        self.scratch_model = ScratchModelEngine()


    def run_model(self, question, config):
        model_type = config["model_type"]           # custom, or openai
        search_type = config["search_type"]         # semantic, semantic_rerank, or typesense
        use_classifier = config["use_classifier"]   # True or False
        n_results = config["n_results"]             # (Cannot be 1) Number of RAG results to return
        model_kwargs = config["model_kwargs"]       # kwargs for model (temperature, max_tokens, etc.)
        

        # ----- Text Retrieval -----
        if search_type == "semantic":
            context = "Not implemented yet"
            # context = self.text_retriever.retrieve(question, top_n=n_results)
        elif search_type == "semantic_rerank":
            context = self.text_retriever.retrieve(
                question, 
                top_n=n_results,
                use_classifier=use_classifier
            )
        elif search_type == "typesense":
            context = self.typesense_retriever.retrieve(
                question, 
                top_n=n_results,
                use_classifier=use_classifier
            )

        print("-"*150)
        print(colored(f"Search Type: {search_type}", "green"))
        print(colored(f"Question: {question}", "blue"))
        print(colored(f"Context: {context}", "yellow"))


        # ----- Language Generation -----
        if model_type == "custom":
            response = self.finetuned_model(question, **model_kwargs)
            for result in response:
                yield result

        elif model_type == "openai":
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": openai_system},
                    {"role": "user", "content": openai_prompt(context, question)},
                ],
                **model_kwargs,
                stream=True,
            )

            message = ""
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content is not None:
                    message += content
                    yield message


question_client = Manage_QA()

def rag(question, history, search_method, model_type, use_classifier, max_results):

    config = {
        "model_type": model_type,
        "search_type": search_method,
        "use_classifier": use_classifier,
        "n_results": max_results,
        "model_kwargs": {
            "temperature": 0.9
        }
    }

    for result in question_client.run_model(question, config):
        yield result

import signal
import sys

def signal_handler(sig, frame):
    print(colored("\n\nInterrupt received, cleaning up...", "red"))

    global question_client
    del question_client

    tf.keras.backend.clear_session()
    
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


demo = gr.ChatInterface(
    rag,
    chatbot=gr.Chatbot(
        height="68vh",
        avatar_images=("./user.png", "./BrockportGPT Logo.png"),
        bubble_full_width=False,
    ),
    additional_inputs=[
        gr.Dropdown(
            choices=["semantic", "semantic_rerank", "typesense"],
            value="typesense",
            label="Search Method",
        ),
        gr.Dropdown(
            choices=["openai", "custom"],
            value="openai",
            label="Model Type"
        ),
        gr.Checkbox(value=True, label="Use Classifier"),
        gr.Slider(2, 6, render=False),
    ],
)

demo.launch(show_api=False, inbrowser=True)