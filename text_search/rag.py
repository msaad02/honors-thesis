"""
Script containing logic to retrieve text and answer questions using RAG.

Currently only supports GPT-3.5 and GPT-4 for language modeling, but can be
expanded (and likely will be) to support finetuned models.

This class uses all the available retrievers with and without 
question topic classification including:

1. Semantic Search
2. Semantic Reranking
3. Typesense
"""

import os
from openai import OpenAI
from text_search.text_retriever_class import TypesenseRetriever, TextRetriever
from torch.cuda import is_available
from termcolor import colored
import types

openai_system = """You are a helpful chatbot for SUNY Brockport who answers questions using the context given. Be enthusiastic, straightforward, brief, and happy to help in your responses. In general, prefer to be give broad answers unless the question is asking for details. Do not answer questions unrelated to SUNY Brockport. If the answer is not clear from the context, say "I'm sorry, I don't know"."""

openai_prompt = lambda context, question: f"Context: {context}\n\nQuestion: {question}"

class RAG:
    "Wrapper class for RAG retrievers and answerers using `__call__` method."
    def __init__(
        self,
        main_categorization_model_dir: str = "./models/main_category",
        subcategorization_model_dir: str = "./models/subcategory_models/",
        embeddings_file: str = "./data/embeddings.pickle",
        typesense_host: str = "localhost",
        typesense_port: str = "8108",
        typesense_protocol: str = "http",
        typesense_collection_name: str = "brockport_data_v1",
        typesense_api_key: str = "xyz",
        device: str = 'cuda' if is_available() else 'cpu',
        openai_api_key: str = None,
        category_logging: bool = False, # Print categories to console or not
    ):
        # For typesense retrieval
        self.typesense_retriever = TypesenseRetriever(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
            typesense_host=typesense_host,
            typesense_port=typesense_port,
            typesense_protocol=typesense_protocol,
            typesense_collection_name=typesense_collection_name,
            typesense_api_key=typesense_api_key,
            print_categories=category_logging # For logging purposes
        )

        # Does semantic search AND semantic/rerank search
        self.semantic_rerank_retriever = TextRetriever(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
            embeddings_file=embeddings_file,
            device=device,
            print_categories=category_logging # For logging purposes
        )

        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        self.openai_client = OpenAI(api_key=openai_api_key)

    def get_context(
        self, 
        question: str, 
        config: dict, 
        verbose: bool = False
    ) -> str:
        "Gets the context for the question using the specified search type."

        # model_type = config["model_type"]  # finetuned, or openai
        search_type = config["search_type"]  # semantic, semantic_rerank, or typesense
        use_classifier = config["use_classifier"]  # True or False
        n_results = config["n_results"]  # Number of RAG results to return
        
        if verbose:
            print("-" * 150)
            print(colored(f"Search Type: {search_type}", "green"))
            print(colored(f"Question: {question}", "blue"))

        if search_type == "semantic":
            # Semantic search. This uses the same class as reranker but different option
            context = self.semantic_rerank_retriever.retrieve(
                question, 
                top_n=n_results, 
                use_classifier=use_classifier,
                type="semantic"
            )
        elif search_type == "semantic_rerank":
            # Search with semantic search and reranker
            context = self.semantic_rerank_retriever.retrieve(
                question, 
                top_n=n_results, 
                use_classifier=use_classifier,
                type="semantic_rerank"
            )
        elif search_type == "typesense":
            # Search with typesense (hybrid keyword/semantic search)
            context = self.typesense_retriever.retrieve(
                question, 
                top_n=n_results, 
                use_classifier=use_classifier
            )
        elif search_type == "none":
            context = "No search method selected."
        else:
            raise ValueError(f"Invalid search type: {search_type}")

        if verbose:
            print(colored(f"Context: {context}", "yellow"))

        return context


    def use_openai(
        self, 
        question: str, 
        context: str, 
        model_kwargs: dict, 
        stream: bool = False
    ) -> str:
        "Uses OpenAI to answer the question."

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": openai_system},
                {"role": "user", "content": openai_prompt(context, question)},
            ],
            **model_kwargs,
            stream=stream,
        )

        # Python generator objects have weird behavior when returning. This is
        # the only way I could get it to work.
        if stream:
            return self._stream_result(response)
        else:
            return self._return_result(response)
        

    def _stream_result(self, generator: types.GeneratorType):
        "Streams the result from the generator."

        message = ""
        for chunk in generator:
            content = chunk.choices[0].delta.content
            if content is not None:
                message += content
                yield message


    def _return_result(self, response) -> str:
        "Returns the result from the response."

        return response.choices[0].message.content


    def __call__(
        self,
        question: str,
        config: dict,
        stream: bool = False,
        verbose: bool = False
    ) -> str:
        "Uses RAG to answer the question."

        # Checking to make sure the config is valid
        try:
            # model_type = config["model_type"]
            search_type = config["search_type"]
            use_classifier = config["use_classifier"]
            n_results = config["n_results"]
        except KeyError:
            raise KeyError("Invalid config.")

        assert all([
            # model_type in ["finetuned", "openai"],
            search_type in ["semantic", "semantic_rerank", "typesense", "none"],
            isinstance(use_classifier, bool),
            isinstance(n_results, int),
        ])

        # Getting the context
        context = self.get_context(question, config, verbose=verbose)

        # Getting the answer
        return self.use_openai(
            question,
            context,
            model_kwargs=config["model_kwargs"],
            stream=stream
        )