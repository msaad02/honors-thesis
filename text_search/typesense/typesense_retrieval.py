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
                prediction['sub_probs'] = {f'{category}-{subcat}': prob for subcat, prob in sub_probs.items()}
            else:
                prediction['subcategory'] = subcategory_classifier.predict(question)
        return prediction
    
    def _get_text_retrieval_places(self, question: str):
        """
        High level interface between the classifier and the user. Tells us where to do 
        text retrieval based on the probability output of the categorization models.

        It does this by returning the top categories with confidence  <0.2 difference from 
        the highest probability category. (I refer to confidence as the model's probability output.)

        If the model outputted a max confidence of <0.5, then it returns all categories
        to stay safe. This was chosen because the classifier works best when it is confident
        in my experience, and when it is not confident there is usually a grey area.

        Returns:
            dict: {
                'main_categories': [str],
                'subcategories': [str]
            }
        """
        prediction = self._classify_question(question, True)

        # main category
        main_cat_probs_df = pd.DataFrame(
            [(category, prob) for category, prob in prediction['main_probs'].items()], 
            columns=['category', 'probability']
        ).sort_values(by='probability', ascending=False).reset_index(drop=True)

        # Highest category probability
        max_main_prob = main_cat_probs_df['probability'][0]

        # if max_main_prob < 0.5: use everything regardless (classifier is not confident enough)
        if max_main_prob < 0.5:
            main_categories_to_use = main_cat_probs_df['category'].tolist()
            subcategories_to_use = list(self.subcategory_classifiers.keys())
        else:
            # Use all categories at the top within 0.2 of the best category
            main_categories_to_use = main_cat_probs_df[main_cat_probs_df['probability'] > max_main_prob - 0.2]['category'].tolist()

            if 'sub_probs' in prediction.keys():
                subcategory_probs_df = pd.DataFrame(
                    [(category, prob) for category, prob in prediction['sub_probs'].items()], 
                    columns=['category', 'probability']
                ).sort_values(by='probability', ascending=False).reset_index(drop=True)

                # Highest subcategory probability
                max_sub_prob = subcategory_probs_df['probability'][0]

                # Subcategories within 0.2 of the highest subcategory
                subcategories_to_use = subcategory_probs_df[subcategory_probs_df['probability'] > max_sub_prob - 0.2]['category'].tolist()

        text_retreival_places = {
            'main_categories': main_categories_to_use,
            'subcategories': subcategories_to_use if 'sub_probs' in prediction.keys() else []
        }

        return text_retreival_places