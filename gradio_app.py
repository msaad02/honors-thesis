from text_search.text_retriever_class import TextRetriever
import time
import gradio as gr
from openai import OpenAI

client = OpenAI()

text_retriever = TextRetriever(
    main_categorization_model_dir="./text_search/models/main_category_model",
    subcategorization_model_dir="./text_search/models/subcategory_models/",
    embeddings_file="./text_search/data/embeddings.pickle",
)

system = """You are a helpful chatbot for SUNY Brockport who answers questions using the context given. Be enthusiastic, straightforward, and brief in your responses. Do not answer questions unrelated to SUNY Brockport. If the answer is not clear from the context, say "I'm sorry, I don't know"."""

prompt = lambda context, question: f"Context: {context}\n\nQuestion: {question}"

def slow_echo(message, history, one, three):#, text, text2, slider):
    response = text_retriever.retrieve(message)
    for i in range(len(response)):
        time.sleep(0.01)
        yield response[: i+1]

def rag(question, history, search_method, max_results):
    context = text_retriever.retrieve(question, top_n=max_results)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt(context, question)},
        ],
        temperature=1,
        stream=True
    )

    message = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is not None:
            message += content
            yield message


demo = gr.ChatInterface(
    rag, 
    additional_inputs=[
        gr.Dropdown([
            "Classifier - Semantic - Reranking",
            "Classifier - Typesense",
            "Semantic"
        ], value="Classifier - Semantic - Reranking", label="Search Method"),
        gr.Slider(1, 10, render=False)
    ]
).launch()