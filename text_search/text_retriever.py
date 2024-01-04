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
NOTE: For each function there is a metadata parameter. If set to True, the function will return
a tuple with the output and a dictionary containing metadata about all the previous function 
results. This is useful for debugging and understanding the output of the function. If set to 
False, the function will only return the output of the current step, which is the default behavior.
-----------------------------------------------------------------------------------------------
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from question_classifier import QuestionClassifier
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd
import pickle
import os

class TextRetriever:
    "Contains all the logic for text retrieval and reranking to enable intelligent text retrieval"
    def __init__(
            self, 
            main_categorization_model_dir: str = "model",
            subcategorization_model_dir: str = "subcat_models/",
            embeddings_file: str = "embeddings.pickle",
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            question_classifier: bool = True
        ):
        self.categorized_data = load_dataset("msaad02/categorized-data", split="train").to_pandas()
        self.device = device
        self.question_classifier = question_classifier

        # Categorization
        self.main_classifier = QuestionClassifier(model_dir=main_categorization_model_dir)
        self.subcategory_classifiers = {}
        for subcat in os.listdir(subcategorization_model_dir):
            self.subcategory_classifiers[subcat] = QuestionClassifier(subcategorization_model_dir + subcat)

        # Get embeddings and model
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        embeddings = pickle.load(open(embeddings_file, "rb"))
        self.embeddings = embeddings['embeddings']
        self.data = embeddings['data']

        # Reranker
        self.rerank_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large').to(self.device)
        self.rerank_model.eval()

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
    
    def _get_text_retrieval_places(self, question: str, metadata: bool = False):
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
        if max_main_prob < 0.5 or not self.question_classifier:
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

        if metadata:
            metadata_dict = {"question": question, "prediction": prediction, "text_retreival_places": text_retreival_places}
            return text_retreival_places, metadata_dict
        else:
            return text_retreival_places

    def _semantic_search(self, question: str, top_n: int = 25, metadata: bool = False, callback_all_categories: bool = False):
        """
        This uses the output from `_get_text_retrieval_places` to do semantic search on the text.

        It returns the `top_n` results from the semantic search. The output of this function is a dataframe
        containing the text and the semantic similarity score - higher is better.
        """
        if not callback_all_categories:
            text_retrieval_places = self._get_text_retrieval_places(question, metadata=metadata)
        else:
            text_retrieval_places = self._get_text_retrieval_places("", metadata=metadata)

        if metadata:
            text_retrieval_places, metadata_dict = text_retrieval_places

        instruction = "Represent this sentence for searching relevant passages: "
        question_embedding = self.embedding_model.encode([instruction + question], normalize_embeddings=True)

        text_embedding_for_question = []
        raw_text_for_question = []

        for category in text_retrieval_places['main_categories']:
            if category in self.embeddings.keys():
                text_embedding_for_question.extend(self.embeddings[category])
                raw_text_for_question.extend(self.data[category])

        for subcategory in text_retrieval_places['subcategories']:
            if subcategory in self.embeddings.keys():
                text_embedding_for_question.extend(self.embeddings[subcategory])
                raw_text_for_question.extend(self.data[subcategory])

        try:
            corpus_embeddings = np.stack(text_embedding_for_question)
        except:
            Warning("Category error. Re-do with all categories.")
            return self._semantic_search(question, top_n, metadata, callback_all_categories=True)

        similarity = question_embedding @ corpus_embeddings.T
        top_args = similarity[0].argsort()[::-1][:top_n]

        data = pd.DataFrame({
            "text": pd.Series(raw_text_for_question)[top_args],
            "semantic_similarity": similarity[0][top_args]
        }).sort_values(by='semantic_similarity', ascending=False).reset_index(drop=True)

        if metadata:
            metadata_dict["similarity_scores"] = data.to_json()
            return data, metadata_dict
        else:
            return data

    def _rerank(self, question: str, top_n: int = 25, metadata: bool = False):
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
        matches = self._semantic_search(question, top_n, metadata=metadata)

        if metadata:
            matches, metadata_dict = matches

        pairs = [[question, text] for text in matches['text'].to_list()]

        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(self.device)
            
            matches['rerank_similarity'] = self.rerank_model(**inputs, return_dict=True).logits.view(-1,).float().cpu()
            matches = matches.sort_values(by='rerank_similarity', ascending=False).reset_index(drop=True)

        if metadata:
            metadata_dict["similarity_scores"] = matches.to_json()
            return matches, metadata_dict
        else:
            return matches
    
    def retrieve(
            self, 
            question: str, 
            top_n: int = 3,
            join_char: str = "\n\n",
            top_n_semantic: int = 25, 
            metadata: bool = False
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
        """

        rerank = self._rerank(question, top_n_semantic, metadata=metadata)

        if metadata:
            rerank, metadata_dict = rerank

        rerank = rerank.head(top_n)
        text = join_char.join(rerank['text'].to_list())

        if metadata:
            metadata_dict["text"] = text
            return text, metadata_dict
        else:
            return text
    