"""
(Maybe) The be all end all of the project.

We will use the search engine from `text_search/text_retriever.py` to retrieve and GPT-4 to generate.
"""
from text_retriever import TextRetriever
from typesense_test.typesense_retrieval import TypesenseRetrieval
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
        self.retriever = TypesenseRetrieval(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
        #    embeddings_file=embeddings_file
        )
        self.client = OpenAI()

        self.system = """You are a helpful chatbot for SUNY Brockport who answers questions using the context given. Be enthusiastic, straightforward, and brief in your responses. Do not answer questions unrelated to SUNY Brockport. If the answer is not clear from the context, say "I'm sorry, I don't know".""" if system_prompt is None else system_prompt

        self.prompt = lambda context, question: f"Context: {context}\n\nQuestion: {question}" if prompt is None else prompt


    def generate(self, question, return_context: bool=False):
        """
        Generate a response to a query.

        Args:
            query (str): The query to respond to.
            return_context (bool, optional): Whether or not to return the context. Defaults to False.

        Returns:
            str: The response to the query.
        """
        context = self.retriever.ask(question)

        response = self.client.chat.completions.create(
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

        if return_context:
            return response.model_dump()['choices'][0]['message']['content'], context
        else:
            return response.model_dump()['choices'][0]['message']['content']