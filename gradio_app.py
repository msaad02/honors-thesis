"""
Contains code to oversee all components of QA via "Manage_QA" class, and
launches the Gradio UI which allows users to interact with the QA system.
"""

from text_search.text_retriever_class import TextRetriever, TypesenseRetriever
import gradio as gr
from openai import OpenAI
from termcolor import colored
from torch.cuda import is_available
from typing import Optional
from scratch_model.inference import ScratchModel
from fine_tuning.finetune_class import FineTunedEngine
import tensorflow as tf
import docker
import signal
import sys
import subprocess

TYPESENSE_CONTAINER_NAME = "typesense_container"

# NOTE TO SELF:
# * Semantic retrieval NOT IMPLEMENTED.
# * Typesesse retrieval is USING CLASSIFIER regardless of checkbox value.

openai_system = """You are a helpful chatbot for SUNY Brockport who answers questions using the context given. Be enthusiastic, straightforward, brief, and happy to help in your responses. In general, prefer to be give broad answers unless the question is asking for details. Do not answer questions unrelated to SUNY Brockport. If the answer is not clear from the context, say "I'm sorry, I don't know"."""

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
        openai_api_key: Optional[str] = None,
    ):
        print("Starting typesense docker container...")
        self.start_docker_container()

        print(colored("Loading Models...\n", color="red", attrs=["bold"]))
        if device is None:
            device = "cuda" if is_available() else "cpu"
            model_type = "gptq" if device == "cuda" else "gguf"

        self.text_retriever = TextRetriever(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
            embeddings_file=embeddings_file,
            question_classifier=use_classifier,
            device=device,
        )

        self.typesense_retriever = TypesenseRetriever(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
            typesense_host=typesense_host,
            typesense_port=typesense_port,
            typesense_protocol=typesense_protocol,
            typesense_collection_name=typesense_collection_name,
            typesense_api_key=typesense_api_key,
            question_classifier=use_classifier,
        )

        # self.semantic_retriever = SemanticRetriever()

        self.openai_client = OpenAI(api_key=openai_api_key)

        self.finetuned_model = FineTunedEngine(
            model_name=finetuned_model_name, model_type=model_type, stream=True
        )

        self.scratch_model = ScratchModel(
            model_dir="./scratch_model/models/transformer_v5/"
        )

    def start_docker_container(self):
        "Starts up typesense docker container if it is not already running."
        client = docker.from_env()

        start = True
        try:
            container = client.containers.get(TYPESENSE_CONTAINER_NAME)
            if container.status == "running":
                print("Docker container is already running...")
                start = False

        except docker.errors.NotFound:
            print("Docker container not found...")

        if start:
            print("Docker container is not running. Starting it now...")
            command = (
                f"docker run --name {TYPESENSE_CONTAINER_NAME} -p 8108:8108 "
                "-v /home/msaad/typesense-data:/data "
                "typesense/typesense:0.25.2 "
                "--data-dir /data "
                "--api-key=xyz "
                "--enable-cors"
            )
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True,
            )
            print("Docker container started.\n")
            

    def get_context(self, question, config):
        "Gets the context for the question using the specified search type."

        model_type = config["model_type"]  # finetuned, or openai
        search_type = config["search_type"]  # semantic, semantic_rerank, or typesense
        use_classifier = config["use_classifier"]  # True or False
        n_results = config["n_results"]  # (Cannot be 1) Number of RAG results to return
        
        if model_type == "openai":
            if search_type == "none":
                context = "No search method selected."
            elif search_type == "semantic":
                context = "Not implemented yet"
                # context = self.text_retriever.retrieve(question, top_n=n_results)
            elif search_type == "semantic_rerank":
                context = self.text_retriever.retrieve(
                    question, top_n=n_results, use_classifier=use_classifier
                )
            elif search_type == "typesense":
                context = self.typesense_retriever.retrieve(
                    question, top_n=n_results, use_classifier=use_classifier
                ) 
        else:
            context = "N/A"

        print("-" * 150)
        print(colored(f"Search Type: {search_type}", "green"))
        print(colored(f"Question: {question}", "blue"))
        print(colored(f"Context: {context}", "yellow"))

        return context


    def get_answer(self, question, context, config):
        "Gets the answer to the question using the specified model type."

        model_type = config["model_type"]
        model_kwargs = config["model_kwargs"]

        if model_type == "finetuned":
            response = self.finetuned_model(question, **model_kwargs)
            for result in response:
                yield result

        elif model_type == "scratch":
            response = self.scratch_model(question, stream=True)
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


    def run_model(self, question, config):
        "Runs the QA system and yields the results."

        try: # ----- Get Context -----
            context = self.get_context(question, config)
        except:
            context = "Error getting context."

        try: # ----- Language Generation -----
            response = self.get_answer(question, context, config)
            for result in response:
                yield result
        except:
            yield "Error generating response."


question_client = Manage_QA()

def rag(question, history, model_type, search_method, use_classifier, max_results):
    config = {
        "model_type": model_type,
        "search_type": search_method,
        "use_classifier": use_classifier,
        "n_results": max_results,
        "model_kwargs": {"temperature": 0.9},
    }

    for result in question_client.run_model(question, config):
        yield result


demo = gr.ChatInterface(
    rag,
    chatbot=gr.Chatbot(
        height="68vh",
        avatar_images=("./user_logo.png", "./brockportgpt_logo.png"),
        bubble_full_width=False,
    ),
    additional_inputs=[
        gr.Dropdown(choices=["openai", "finetuned", "scratch"], value="openai", label="Model Type"),
        gr.Dropdown(
            choices=["typesense", "semantic_rerank", "semantic", "none"],
            value="typesense",
            label="Search Method",
        ),
        gr.Checkbox(value=True, label="Search Method - Use Classifier"),
        gr.Slider(1, 6, value=3, step=1),
    ],
)


def signal_handler(sig, frame):
    "Clean up when the program is interrupted."

    print(colored("\n\nInterrupt received, cleaning up...", "red"))

    global question_client
    del question_client

    tf.keras.backend.clear_session()

    client = docker.from_env()
    try:
        container = client.containers.get(TYPESENSE_CONTAINER_NAME)
        container.stop()
        container.remove()
        print(f"Container '{TYPESENSE_CONTAINER_NAME}' has been stopped.")
    except docker.errors.NotFound:
        print(f"Container '{TYPESENSE_CONTAINER_NAME}' not found.")

    sys.exit(0)


def main():
    "Main function to run the QA system."

    # Register signal handler to clean up when the program is interrupted.
    signal.signal(signal.SIGINT, signal_handler)

    # Launch the Gradio UI.
    demo.launch(show_api=False, inbrowser=True)


if __name__ == "__main__":
    main()