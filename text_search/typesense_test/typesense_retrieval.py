"""
The other retrieval methods in "text_retriever.py" as more manual. They also don't
give amazing results. This is an attempt at a different approach, using Typesense.

On it's own, typesense is okay. It's not great, but seems to be much better than
semantic search on its own. In our case, we are using Typesense as a hybrid search
engine. That means it is using both keyword search and semantic search to find the
best results. 

This script aims to combine the two methods. It will implement the classifier from
previous scripts to initially filter the results. Then, it will use Typesense to
saerch from the filtered results. This should give us the best of both worlds.

Results TBD
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from question_classifier import QuestionClassifier
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import pickle
import os
import typesense

class TypesenseRetrieval:
    def __init__(
            self,
            typesense_host: str = "localhost",
            typesense_port: str = "8108",
            typesense_protocol: str = "http",
            typesense_collection_name: str = "brockport_data_v1",
            typesense_api_key: str = "xyz",
            main_categorization_model_dir: str = "../model",
            subcategorization_model_dir: str = "../subcat_models/"
        ):
        """
        Initialize the TypesenseRetrieval object. This will create a connection to
        the Typesense server, and load the question classifier model.

        I've been using the Typesense docker image, so defaults are set to that.
        """
        self.client = typesense.Client({
            'nodes': [{
                'host': typesense_host,
                'port': typesense_port,
                'protocol': typesense_protocol
            }],
            'api_key': typesense_api_key,
            'connection_timeout_seconds': 60,
            'retry_interval_seconds': 5
        })

        self.collection_name = typesense_collection_name    
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Categorization
        self.main_classifier = QuestionClassifier(model_dir=main_categorization_model_dir)
        self.subcategory_classifiers = {}
        for subcat in os.listdir(subcategorization_model_dir):
            self.subcategory_classifiers[subcat] = QuestionClassifier(subcategorization_model_dir + subcat)

    def _classify_question(self, question: str, return_probabilities: bool = False):
        """
        Classifies user question into a main category and subcategory (if applicable).

        There is no other processing done in this step, it returns a dictionary with relevant
        information for the next step, and an option to return the probabilities of the model
        to better understand the model's confidence in its prediction.
        """
        prediction = {}
        if return_probabilities:
            prediction['category'], prediction['main_probs'] = self.main_classifier.predict(question, True)
        else:
            prediction['category'] = self.main_classifier.predict(question)

        category = prediction['category']
        if category in self.subcategory_classifiers:
            subcategory_classifier = self.subcategory_classifiers[category]

            if return_probabilities:
                prediction['subcategory'], sub_probs = subcategory_classifier.predict(question, True)
                prediction['sub_probs'] = {f'{category}|{subcat}': prob for subcat, prob in sub_probs.items()}
            else:
                prediction['subcategory'] = subcategory_classifier.predict(question)
        return prediction

    def _select_text_retrieval_categories(self, question: str, return_probabilities: bool = False) -> dict:
        """
        High level interface between the classifier and the user. Tells us where to do 
        text retrieval based on the probability output of the categorization models.

        So this is just a simplifed, more robust version of what is in `../text_retriever.py`.

        We will pick all categories/subcategories with confidence > 0.2. If the main category
        is less than 0.5, then we will use all categories.

        Args:
            question (str): The question to classify
            return_probabilities (bool, optional): Whether to return the probabilities of the model. Defaults to False.
        """
        prediction = self._classify_question(question, return_probabilities=True)

        main_category_scores = (
            pd.DataFrame(prediction['main_probs'].items(), columns=['category', 'score'])
            .sort_values('score', ascending=False)
            .reset_index(drop=True)
        )

        if main_category_scores['score'][0] < 0.3:
            main_categories_to_use = main_category_scores['category'].to_list()
            subcategories_to_use = [] # Uses all if []
        else:
            main_categories_to_use = main_category_scores[main_category_scores['score'] > 0.3]['category'].to_list()

            if 'sub_probs' in prediction.keys():
                subcategory_scores = (
                    pd.DataFrame(prediction['sub_probs'].items(), columns=['category', 'score'])
                    .sort_values('score', ascending=False)
                    .reset_index(drop=True)
                )
                subcategories_to_use = subcategory_scores[subcategory_scores['score'] > 0.15]['category'].to_list()
            else:
                subcategories_to_use = [] # Uses all if []

        if return_probabilities:
            return {
                'main_categories': main_categories_to_use,
                'sub_categories': subcategories_to_use,
                'main_probs': prediction['main_probs'],
                'sub_probs': prediction['sub_probs'] if 'sub_probs' in prediction.keys() else {}
            }
        else:
            return {
                'main_categories': main_categories_to_use,
                'sub_categories': subcategories_to_use
            }

    def _combine_categorization_with_search(self, question, alpha=0.8):
        """
        End to end function that takes in a question and returns the results of a hybrid search
        using the question and the categories that the question was classified into.

        Parameters
        ----------
        question : str
            The question to be answered
        alpha : float
            The weighting parameter for the hybrid search. Higher = more weight on semantic search,
            lower = more weight on keyword search
        """
        categories = self._select_text_retrieval_categories(question)

        filter_by_query = f'main_category: {str(categories["main_categories"])}'

        if categories['sub_categories'] != []:
            categories['sub_categories'] = [subcat.split("|")[1] for subcat in categories['sub_categories']]
        
            filter_by_query = f'main_category: {str(categories["main_categories"])} && sub_category: {str(categories["sub_categories"])}'
        
        query = {
            'q': question,
            'filter_by': filter_by_query
        }

        response = self.client.multi_search.perform(
            search_queries={'searches': [query]},
            common_params = {
                'collection': 'brockport_data_v1',
                'query_by': 'embedding,context',
                'limit': 2,
                'prefix': False,
                'vector_query': f'embedding:([], alpha: {alpha})',        
                'exclude_fields': 'embedding'
            }
        )

        return {
            'question': question,
            'categories': categories,
            'response': response
        }

    def ask(self, question: str, alpha: float = 0.8):
        """
        End to end function that takes in a question and returns the results of a hybrid search
        using the question and the categories that the question was classified into.

        Parameters
        ----------
        question : str
            The question to be answered
        alpha : float
            The weighting parameter for the hybrid search. Higher = more weight on semantic search,
            lower = more weight on keyword search
        """
        search = self._combine_categorization_with_search(question, alpha)

        # Pull out the raw text from the results
        results = [x['document']['context'] for x in search['response']['results'][0]['hits']]

        # Join the results together
        results = "\n\n".join(results)

        return results