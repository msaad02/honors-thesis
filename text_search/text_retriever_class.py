"""
The following contains all our text retrieval models and functions. Each method is
split into its own class, and each class has its own documentation. The classes are:

1. TypesenseRetriever: Uses categorization and Typesense for hybrid search
2. TextRetriever: Uses categorization, semantic search and reranking for text retrieval

Each of these classes have slightly different dependencies and interfaces. To get any
working you need to run the setup scripts specified in their descriptions. Alternatively,
if you run all the setup scripts (1-5 in order) you should be able to run all the classes
without any issues, provided typesense is running and you have a valid OPENAI_API_KEY 
environment variable set.

Each class has a `retrieve` function that takes in a question and returns the results.

Note that the TextRetriever class doubles as a pure semantic search retriever
since it has the option to not use the reranker.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from category_classifier import QuestionClassifier
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import typesense
import torch
import docker
import subprocess

TYPESENSE_CONTAINER_NAME = "typesense_container"

# ----------------------------------------------------------------------------------------------------------------------

class TypesenseRetriever:
    """
    The other retrieval methods in "text_retriever.py" as more manual. They also don't
    give amazing results. This is an attempt at a different approach, using Typesense.

    On it's own, typesense is okay. It's not great, but seems to be much better than
    semantic search on its own. In our case, we are using Typesense as a hybrid search
    engine. That means it is using both keyword search and semantic search to find the
    best results.

    This script aims to combine all the methods. It will implement the classifier to 
    initially filter the results. Then, it will use Typesense hybrid search from the 
    filtered results. This should give us the best of both worlds.
    """

    def __init__(
        self,
        main_categorization_model_dir: str = "./models/main_category_model",
        subcategorization_model_dir: str = "./models/subcategory_models/",
        typesense_host: str = "localhost",
        typesense_port: str = "8108",
        typesense_protocol: str = "http",
        typesense_collection_name: str = "brockport_data_v1",
        typesense_api_key: str = "xyz",
        print_categories: bool = False
    ):
        """
        Initialize the TypesenseRetrieval object. This will create a connection to
        the Typesense server, and load the question classifier model.

        I've been using the Typesense docker image, so defaults are set to that.
        """
        # Start the typesense docker container if it is not already running

        # NOTE: This is helpful automation for me, the author, but I suspect it will be a breaking
        # issue for other people. If you don't want this, just manually setup your typesense server
        # via docker and then set the typesense_host, typesense_port, typesense_protocol, and
        # typesense_api_key parameters to the correct values. Then COMMENT OUT the following line!
        # There is a similar line in the `__del__` function. Make sure to comment that out too!
        self.client = docker.from_env()
        self.start_docker_container()

        self.typesense_client = typesense.Client(
            {
                "nodes": [
                    {
                        "host": typesense_host,
                        "port": typesense_port,
                        "protocol": typesense_protocol,
                    }
                ],
                "api_key": typesense_api_key,
                "connection_timeout_seconds": 60,
                "retry_interval_seconds": 5,
            }
        )

        self.collection_name = typesense_collection_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self.categorizer = QuestionClassifier(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
        )

        # Just included so in the app we can see what categories the question was classified into
        self.print_categories = print_categories


    def start_docker_container(self):
        """
        Starts up typesense docker container if it is not already running.

        NOTE: This is helpful automation for me, the author, but I suspect its a breaking
        issue for other people. If you don't want this, comment out the start_docker_container
        line, and remove the stop_docker_container function call in __del__(). Instead, you can
        just manually setup your typesense server via docker and set the parameters in the 
        `__init__` function to the correct values.
        """
        try:
            self.container = self.client.containers.run(
                image="typesense/typesense:0.25.2",
                name=TYPESENSE_CONTAINER_NAME,
                ports={'8108/tcp': 8108},
                volumes={'/home/msaad/typesense-data': {'bind': '/data', 'mode': 'rw'}},
                command="--data-dir /data --api-key=xyz --enable-cors",
                detach=True
            )
            print("Typesense container started.\n")
        except docker.errors.APIError:
            print("Typesense container (presumably) exists already.")

            self.container = self.client.containers.get(TYPESENSE_CONTAINER_NAME)

            if self.container.status != "running":
                self.container.start()
                print("Typesense container started.\n")
            else:
                print("Typesense container is already running...")

        except Exception as e:
            print(f"Error starting Typesense container: \n\n{e}")


    def stop_docker_container(self):
        "Stops the typesense docker container if it is running."
        try:
            self.container.stop()
            self.container.remove()
        except:
            print(f"There was an error shutting down the typesense container, please manually remove.")


    def _search(self, question, alpha=0.8, use_classifier=True):
        """
        End to end function that takes in a question and returns the results of a hybrid search
        using the question and the categories that the question was classified into.

        Parameters
        ----------
        question : str
            The question to be answered
        top_n : int
            The number of results to return
        alpha : float
            The weighting parameter for the hybrid search. Higher = more weight on semantic search,
            lower = more weight on keyword search
        use_classifier : bool
            Whether or not to use the classifier to filter the results. Recommended to be True.
        """
        if use_classifier:
            categories = self.categorizer.predict(question)
            if self.print_categories:
                print(categories)

            if categories["sub_categories"] == []:
                # If there are no subcategories, we can just search by main category
                filter_by_query = f'main_category: {str(categories["main_categories"])}'
            else:
                # If there are subcategories, we need to search by both main and subcategories
                categories["sub_categories"] = [
                    subcat.split("|")[1] for subcat in categories["sub_categories"]
                ]
                filter_by_query = f'main_category: {str(categories["main_categories"])} && sub_category: {str(categories["sub_categories"])}'

            query = {"q": question, "filter_by": filter_by_query}
        else:
            # If we aren't using the classifier, just search by the question
            if self.print_categories:
                print("Selected 'no categorization' - continuing with searching all categories...")

            query = {"q": question}

        response = self.typesense_client.multi_search.perform(
            search_queries={"searches": [query]},
            common_params={
                "collection": self.collection_name,
                "query_by": "embedding,context",
                "limit": 10,
                "prefix": False,
                "vector_query": f"embedding:([], alpha: {alpha})",
                "exclude_fields": "embedding",
            },
        )

        if use_classifier:
            return {"question": question, "categories": categories, "response": response}
        else:
            return {"question": question, "response": response}

    def retrieve(
        self, 
        question: str, 
        top_n: int = 3, 
        alpha: float = 0.8, 
        use_classifier: bool = True
    ):
        """
        End to end function that takes in a question and returns the results of a hybrid search
        using the question and the categories that the question was classified into.

        Parameters
        ----------
        question : str
            The question to be answered
        top_n : int
            The number of results to return
        alpha : float
            The weighting parameter for the hybrid search. Higher = more weight on semantic search,
            lower = more weight on keyword search
        use_classifier : bool
            Whether or not to use the classifier to filter the results. Recommended to be True.
        """
        if question == "":
            return "No question given."

        response = self._search(question, alpha, use_classifier)

        if "code" in response["response"]["results"][0].keys():
            return "No results found."

        # Pull out and merge raw text results
        results = response['response']['results'][0]
        results = [x["document"]["context"] for x in results["hits"][0:top_n]]
        results = "\n\n".join(results)

        return results
    
    def __del__(self):
        "When the object is deleted, stop the typesense docker container."
        self.stop_docker_container()


# ----------------------------------------------------------------------------------------------------------------------

class TextRetriever:
    """
    TextRetriever: A Class for Intelligent Text Retrieval and Reranking

    The TextRetriever class is designed to intelligently retrieve and rank text passages
    relevant to a given query, primarily questions. The class integrates several advanced
    NLP techniques and models to achieve high accuracy and relevance in its outputs.

    How it works:
    1. Categorization: Utilizes a main categorization model to classify the question into
    broad categories and subcategories (if applicable). This is achieved using a custom
    QuestionClassifier and models trained in other scripts in this repository.

    2. Text Retrieval: Based on the categorization, it identifies relevant text passages
    from a pre-processed dataset. This step involves semantic search to generate embeddings
    for the question and compare them with the embeddings of the stored text.

    3. Reranking: After retrieving a set of relevant passages, the class employs a reranker
    model, specifically a cross-encoder architecture, to reorder these passages. The reranker
    evaluates the interaction between the question and each text passage to refine the results,
    ensuring that the most relevant passages are ranked higher.

    This class effectively combines categorization, semantic search, and reranking to create a
    comprehensive text retrieval system.

    -----------------------------------------------------------------------------------------------
    NOTE: You can optionally use the classifier to filter the results, and you can also optionally
    use the reranker or not. So there are a total of 4 different ways to use this class:

    1. Use the classifier and the reranker (which uses the classifier + semantic search + reranker model)
    2. Use the classifier and not the reranker (which uses the classifier + ONLY semantic search)
    3. Don't use the classifier and use the reranker (which uses semantic search + reranker model)
    4. Don't use the classifier and don't use the reranker (which uses ONLY semantic search)
    -----------------------------------------------------------------------------------------------
    """

    def __init__(
        self,
        main_categorization_model_dir: str = "./models/main_category_model",
        subcategorization_model_dir: str = "./models/subcategory_models/",
        embeddings_file: str = "./data/embeddings.pickle",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        print_categories: bool = False
    ):
        self.device = device
        self.print_categories = print_categories

        # Categorization
        self.categorizer = QuestionClassifier(
            main_categorization_model_dir=main_categorization_model_dir,
            subcategorization_model_dir=subcategorization_model_dir,
        )

        # Get embeddings and model
        self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.embeddings = pd.read_pickle(embeddings_file)

        # Reranker
        self.rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-large"
        ).to(self.device)
        self.rerank_model.eval()

    def _semantic_search(
        self,
        question: str,
        top_n: int = 25,
        use_classifier: bool = True,
    ):
        """
        This uses the output from `_get_text_retrieval_places` to do semantic search on the text.

        It returns the `top_n` results from the semantic search. The output of this function is a dataframe
        containing the text and the semantic similarity score - higher is better.
        """
        instruction = "Represent this sentence for searching relevant passages: "
        question_embedding = self.embedding_model.encode(
            [instruction + question], normalize_embeddings=True
        )

        if use_classifier:
            categories = self.categorizer.predict(question)
            if self.print_categories:
                print(categories)

            # Building the search list (categories to filter to in the embeddings dataframe)
            if categories['sub_categories'] != []:
                search_list = list(set(categories['main_categories'] + categories['sub_categories']))
            else:
                # This is weird, but since I'm treating 'sub_categories' as ALL if I return [] (empty list), 
                # then I need to actually return all the possible subcategories that are inside the main categories selected.

                # Short explanation: For this retrieval system I'm searching for categories to INCLUDE, not EXCLUDE like in the
                # typesense search for instance, so if I were to return [] (empty list) then I would NOT SEARCH the subcategories,
                # which is not what I want. I want to search ALL subcategories if I return [] (empty list).

                # So, since all "subcategories" are labeled as "main_catgory|sub_category", I can just iterate through the
                # index of the embeddings and check if any of the main categories are in that "main_category" part of the index.
                tf_list = [category in categories['main_categories'] for category in self.embeddings.index.str.split('|').str[0]]
                search_list = list(set(categories['main_categories'] + self.embeddings.loc[tf_list].index.to_list()))

            # Now that I have the search list, I can just filter through it and get the embeddings and raw text
            search_list_fixed = []

            for category in search_list:
                if category in self.embeddings.index:
                    search_list_fixed.append(category)
                        
            embeddings_and_text = self.embeddings.loc[search_list_fixed]

        else:
            if self.print_categories:
                print("Selected 'no categorization' - continuing with searching all categories...")
            
            # If no categories are found, then just search all categories
            embeddings_and_text = self.embeddings

        corpus_embeddings = embeddings_and_text["embeddings"].to_list()
        corpus_text = embeddings_and_text["data"].to_list()

        # Flatten the lists (instead of lists of lists, we want a single big list)
        corpus_embeddings = [item for sublist in corpus_embeddings for item in sublist]
        corpus_text = [item for sublist in corpus_text for item in sublist]

        try:
            corpus_embeddings = np.stack(corpus_embeddings)
        except:
            raise ValueError("No relevant text found for this question.")

        similarity = question_embedding @ corpus_embeddings.T

        data = (
            pd.DataFrame({
                "text": corpus_text,
                "semantic_similarity": similarity[0]
            })
            .sort_values(by="semantic_similarity", ascending=False)
            .drop_duplicates(subset=["text"])
            .reset_index(drop=True)
            .iloc[0:top_n, :]
        )
        return data

    def _rerank(self, question: str, top_n: int = 50, use_classifier: bool = True):
        """
        Pefroms reranking on the top n results of the text retreival step. This uses the classifier
        model and semantic search to filter to a small set of results, then "reranks" them to find the
        true best results.

        The big idea of a reranker is that it is a cross-encoder model. This means that the question
        and the text are encoded together, or that they are dependent on each other. This is different
        than the traditional semantic search model which is a bi-encoder model where the question and
        text are encoded independently. There's just more interaction between the question and the text
        in a cross-encoder model, which improves the results. It is more computationally expensive though
        since it cannot be stored in a database (questions and text are dependent on each other, and questions
        are not known ahead of time). This means that the reranker model has to be run on the fly. That's
        why we use the previous step to filter down to a small set of results, then rerank them.

        top_n is the main parameter that controls how many semantic search results are returned. Less means
        faster results, but potentially worse results. (In reality, the semantic search and reranker are
        correlated pretty well, so the reranker can still do a good job with a small number of results.)
        """
        matches = self._semantic_search(question, top_n, use_classifier=use_classifier)

        pairs = [[question, text] for text in matches["text"].to_list()]

        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            matches["rerank_similarity"] = (
                self.rerank_model(**inputs, return_dict=True)
                .logits.view(-1)
                .float()
                .cpu()
            )
            matches = matches.sort_values(
                by="rerank_similarity", ascending=False
            ).reset_index(drop=True)

        return matches

    def retrieve(
        self,
        question: str,
        top_n: int = 3,
        join_char: str = "\n\n",
        top_n_semantic: int = 50,
        use_classifier: bool = True,
        type: str = "semantic_rerank", # "semantic_rerank" or "semantic"
    ):
        """
        Retrieves the top n results for a given question. This is the main function of the class.

        For better or for worse, this functions in this class execute in a pipeline. This means that
        the output of one function is the input to the next function. This is a good thing because it
        makes the code easy to understand and debug. However, it is also makes the code less flexible.

        **This function just provides a high level interface to the class.**

        Args:
            question (str): The question to retrieve results for
            top_n (int): The number of results to return
            join_char (str): The character to join the results with
            top_n_semantic (int): The number of results to return from the semantic search step (influences search results)
            metadata (bool): Whether to return metadata about the function results
            use_classifier (bool): Whether to use the classifier to filter the results
            type (str): Whether to use the "semantic_rerank or only "semantic" search method
        """
        if question == "":
            return "No question given."
        
        if type == "semantic_rerank":
            results = self._rerank(question, top_n_semantic, use_classifier=use_classifier)
        elif type == "semantic":
            results = self._semantic_search(question, top_n_semantic, use_classifier=use_classifier)
        else:
            raise ValueError("Invalid type. Must be 'semantic_rerank' or 'semantic'.")

        results = results.head(top_n)
        text = join_char.join(results["text"].to_list())

        return text