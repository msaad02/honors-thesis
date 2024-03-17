
from qa_class import Manage_QA
import gradio as gr
from termcolor import colored
import tensorflow as tf
import signal
import sys
import random

# Create the question client
question_client = Manage_QA()

# Get score
model_score = {"modelA": 0, "modelB": 0}

# ------------------------------- HELPER FUNCTIONS -------------------------------
def randomly_pick_2_configs():
    choices = {
        "model_type": ["rag", "finetuned", "scratch"],
        "search_type": ["typesense", "semantic_rerank", "semantic"],
        "use_classifier": [True, False],
        "n_results": [1, 2, 3, 4, 5, 6],
        "temperature": [0.8, 0.9, 1.0]
    }

    first = {
        "model_type": random.choice(choices["model_type"]),
        "search_type": random.choice(choices["search_type"]),
        "use_classifier": random.choice(choices["use_classifier"]),
        "n_results": random.choice(choices["n_results"]),
        "model_kwargs": {"temperature": random.choice(choices["temperature"])}
    }

    second = {
        "model_type": random.choice(choices["model_type"]),
        "search_type": random.choice(choices["search_type"]),
        "use_classifier": random.choice(choices["use_classifier"]),
        "n_results": random.choice(choices["n_results"]),
        "model_kwargs": {"temperature": random.choice(choices["temperature"])}
    }

    while first == second:
        first, second = randomly_pick_2_configs()

    return first, second

# Set user message when the user sends it
def user(user_message, history):
    return "", history + [[user_message, None]]


# ------------------------------- RUNNING MODELS -------------------------------
def run_model_A(history):
    global first_config

    history[-1][1] = ""
    question = history[-1][0]

    for result in question_client.run_model(question, first_config):
        history[-1][1] = result
        yield history


def run_model_B(history):
    global second_config

    history[-1][1] = ""
    question = history[-1][0]

    for result in question_client.run_model(question, second_config):
        history[-1][1] = result
        yield history


# ------------------------------- CHANGING CONFIGS -------------------------------
def change_configs():
    global first_config, second_config
    first_config, second_config = randomly_pick_2_configs()

    return f"""
    ### Left Config
    {second_config}

    ---

    ### Right Config
    {second_config}
    """
    
# ------------------------------- PICKING MODELS -------------------------------
def modelA_Better():
    model_score["modelA"] += 1
    print(colored(f"\n\nModel A is better. Current score: {model_score}", "green"))

    return [[], []]

def modelB_Better():
    model_score["modelB"] += 1
    print(colored(f"\n\nModel B is better. Current score: {model_score}", "green"))

    return [[], []]


# ------------------------------- UI COMPONENTS -------------------------------
with gr.Blocks() as demo:
    with gr.Row():

        with gr.Column():
            chatbot = gr.Chatbot(height="68vh")
            modelA = gr.Button("Model A")
        with gr.Column():
            chatbot2 = gr.Chatbot(height="68vh")
            modelB = gr.Button("Model B")
    

    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    with gr.Accordion("Show Configs"):
        configs = gr.Markdown(change_configs())


    # LEFT SIDE (MODEL A)
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        run_model_A, chatbot, chatbot
    )

    # RIGHT SIDE (MODEL B)
    msg.submit(user, [msg, chatbot2], [msg, chatbot2], queue=False).then(
        run_model_B, chatbot2, chatbot2
    )

    # Clearing the chatbot
    clear.click(lambda: None, None, chatbot, queue=False)
    clear.click(lambda: None, None, chatbot2, queue=False)

    # Picking the better model
    modelA.click(modelA_Better, None, [chatbot, chatbot2], queue=False)
    modelB.click(modelB_Better, None, [chatbot, chatbot2], queue=False)







    # ADD BUTTON TO CHANGE CONFIGS












# ------------------------------- CLEANUP -------------------------------
def signal_handler(sig, frame):
    print(colored("\n\nInterrupt received, cleaning up...", "red"))

    global question_client
    del question_client

    tf.keras.backend.clear_session()
    sys.exit(0)


# ------------------------------- MAIN FUNCTION -------------------------------
if __name__ == "__main__":

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Launch demo
    demo.launch()

