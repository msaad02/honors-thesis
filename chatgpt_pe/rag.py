"""
(Maybe) The be all end all of the project.

We will use the search engine from `text_search/text_retriever.py` to retrieve and GPT-4 to generate.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../text_search')))

from text_retriever import TextRetriever
from openai import OpenAI

class RAG():
    def __init__(
        self, 
        main_categorization_model_dir="../text_search/model",
        subcategorization_model_dir="../text_search/subcat_models/",
        embeddings_file="../text_search/embeddings.pickle",
        system_prompt=None,
        prompt=None
    ):
        "Load in the relevant stuff."
        self.retriever = TextRetriever(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
            embeddings_file=embeddings_file
        )

        if system_prompt is None:
            self.system = """You are a helpful chatbot for SUNY Brockport. You will be given information about questions people have asked about the school. Your job is to parse that information and generate realistic responses to those questions. Be enthusiastic, straightforward, and brief in your responses. Do not answer questions unrelated to SUNY Brockport, or questions that cannot be answered based on the information given."""
        else:
            self.system = system_prompt

        if prompt is None:
            self.prompt = lambda context, question: f"Context: {context}\n\nQuestion: {question}"
        else:
            self.prompt = prompt



    def generate(self, question):
        """
        Generate a response to a query.

        Args:
            query (str): The query to respond to.

        Returns:
            str: The response to the query.
        """
        context = self.retriever.retrieve(question)

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.prompt(context, question)},
            ],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2
        )

        return response