"""
Contains code to oversee all components of QA via "Manage_QA" class, and
launches the Gradio UI which allows users to interact with the QA system.
"""

from qa_class import Manage_QA
import gradio as gr
from termcolor import colored
import tensorflow as tf
import signal
import sys

# Create the question client
question_client = Manage_QA()

def gradio_function(question, history, model_type, search_method, use_classifier, max_results):
    config = {
        "model_type": model_type,
        "search_type": search_method,
        "use_classifier": use_classifier,
        "n_results": max_results,
        "model_kwargs": {"temperature": 0.8},
    }

    for result in question_client.run_model(question, config):
        yield result

# Create the Gradio UI
demo = gr.ChatInterface(
    gradio_function,
    chatbot=gr.Chatbot(
        height="68vh",
        avatar_images=("./user_logo.png", "./brockportgpt_logo.png"),
        bubble_full_width=False,
    ),
    additional_inputs=[
        gr.Dropdown(
            choices=["rag", "finetuned", "scratch"], 
            value="rag", 
            label="Model Type"
        ),
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
    del question_client # .cleanup()

    tf.keras.backend.clear_session()

    sys.exit(0)

def main():
    "Main function to run the QA system."

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Launch the Gradio UI.
    demo.launch(show_api=False, inbrowser=True, share=True)


if __name__ == "__main__":
    main()
