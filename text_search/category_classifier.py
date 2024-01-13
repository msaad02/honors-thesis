"""
This module contains a question classifier that predicts the category of a given question. 
The model is trained inside of `train_question_classifier.py`. The predicted category can 
be obtained by calling the `predict` method of the `RawQuestionClassifier` class, or, for
production use, the `QuestionClassifier` class. The `RawQuestionClassifier` class is a
helper class that can be used to get predictions from the model, while the `QuestionClassifier`
class is a wrapper around the `RawQuestionClassifier` class that does the logic to determine
what categories to return.

Example usage:
>>> from category_classifier import QuestionClassifier
>>> classifier = QuestionClassifier()
>>> classifier.predict('What will I learn in a math major?')
... {'main_categories': ['academics'], 'sub_categories': ['academics|mathematics']}

Note the unconventional structure of the `sub_categories` key. This is because the subcategory
is a combination of the main category and the subcategory, separated by a pipe. This is done
to make it easier to filter the subcategories in the text retrieval step.
"""

import torch
import torch.nn as nn
import pandas as pd
import joblib
import json
import os

class CategoryClassifier(nn.Module):
    "Model Architecture"
    def __init__(self, vocab_size, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class RawQuestionClassifier:
    "Helper class to get predictions from the qa category classifier model"
    def __init__(self, model_dir='model'):
        # check if model directory exists
        if not os.path.isdir(model_dir):
            raise Exception(f'Model directory {model_dir} does not exist.')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_dir)
        self.vectorizer = joblib.load(f'{model_dir}/vectorizer.joblib')
        self.int_to_text = self._load_class_mappings(f'{model_dir}/class_mappings.json')

    def _load_model(self, model_dir):
        with open(f'{model_dir}/hyperparameters.json', 'r') as f:
            hyperparameters = json.load(f)
        model = CategoryClassifier(
            hyperparameters['vocab_size'],
            hyperparameters['hidden_dim'],
            hyperparameters['output_dim'],
            hyperparameters['dropout']
        )
        model.load_state_dict(torch.load(f'{model_dir}/category_classifier_model.pth'))
        model.eval()
        return model.to(self.device)

    def _load_class_mappings(self, path):
        with open(path, 'r') as f:
            mappings = json.load(f)
        return mappings['int_to_text']

    def predict(self, question: str, return_probabilities: bool = False):
        """
        Predicts the category of a given question.

        Args:
            question (str): The input question.
            return_probabilities (bool, optional): Whether to return the predicted category probabilities. 
                Defaults to False.

        Returns:
            str or tuple: The predicted category. If `return_probabilities` is True, a tuple containing 
                the predicted category and a dictionary of category probabilities is returned.

        """
        question_vect = self.vectorizer.transform([question]).toarray()
        question_tensor = torch.tensor(question_vect, dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            output = self.model(question_tensor)
            predicted_index = output.argmax(1).item()

        predicted_category = self.int_to_text[str(predicted_index)]

        if return_probabilities:
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(output).squeeze().tolist()
            prob_dict = {self.int_to_text[str(idx)]: round(prob, 4) for idx, prob in enumerate(probabilities)}
            return predicted_category, prob_dict

        return predicted_category


class QuestionClassifier:
    """
    Helper class to get predictions from the question classifier model. The main differentiator 
    between this class and the `RawQuestionClassifier` class is that this class will retrieve 
    both the question category and the question subcategory, if one exists. From there, it will 
    use the logit scores to determine what categories to return. Essentially, this class is a 
    wrapper around the `RawQuestionClassifier` class that does the logic.
    """
    def __init__(
        self, 
        main_categorization_model_dir='./models/main_category_model/', 
        subcategorization_model_dir='./models/subcategory_models/'
    ):
        # ---- Attempt to initialize models ----
        try:
            self.main_classifier = RawQuestionClassifier(
                model_dir=main_categorization_model_dir
            )
        except:
            raise Exception(f'Could not load main categorization model from {main_categorization_model_dir}')
        
        try:
            self.subcategory_classifiers = {}
            for subcat in os.listdir(subcategorization_model_dir):
                self.subcategory_classifiers[subcat] = RawQuestionClassifier(
                    subcategorization_model_dir + subcat
                )
        except:
            raise Exception(f'Could not load subcategorization models from {subcategorization_model_dir}')
        
    def _classify_question(self, question: str, return_probabilities: bool = True):
        """
        Classifies user question into a main category and subcategory (if applicable).

        There is no other processing done in this step, it returns a dictionary with relevant
        information for the next step, and an option to return the probabilities of the model
        to better understand the model's confidence in its prediction.
        """
        prediction = {}
        if return_probabilities:
            (
                prediction["category"],
                prediction["main_probs"],
            ) = self.main_classifier.predict(question, True)
        else:
            prediction["category"] = self.main_classifier.predict(question)

        category = prediction["category"]
        if category in self.subcategory_classifiers:
            subcategory_classifier = self.subcategory_classifiers[category]

            if return_probabilities:
                prediction["subcategory"], sub_probs = subcategory_classifier.predict(
                    question, True
                )
                prediction["sub_probs"] = {
                    f"{category}|{subcat}": prob for subcat, prob in sub_probs.items()
                }
            else:
                prediction["subcategory"] = subcategory_classifier.predict(question)
        return prediction
    
    def predict(self, question: str, return_probabilities: bool = False):
        """
        High level interface between the classifier and the user. Tells us where to do
        text retrieval based on the probability output of the categorization models.

        We will pick all categories/subcategories with confidence > 0.2. If the main category
        is less than 0.5, then we will use all categories.

        Args:
            question (str): The question to classify
            return_probabilities (bool, optional): Whether to return the probabilities of the model. Defaults to False.
        """
        prediction = self._classify_question(question, return_probabilities=True)

        main_category_scores = (
            pd.DataFrame(
                prediction["main_probs"].items(), columns=["category", "score"]
            )
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        if main_category_scores["score"][0] < 0.3:
            main_categories_to_use = main_category_scores["category"].to_list()
            subcategories_to_use = []  # Uses all if []
        else:
            main_categories_to_use = main_category_scores[
                main_category_scores["score"] > 0.3
            ]["category"].to_list()

            if "sub_probs" in prediction.keys():
                subcategory_scores = (
                    pd.DataFrame(
                        prediction["sub_probs"].items(), columns=["category", "score"]
                    )
                    .sort_values("score", ascending=False)
                    .reset_index(drop=True)
                )
                subcategories_to_use = subcategory_scores[
                    subcategory_scores["score"] > 0.15
                ]["category"].to_list()
            else:
                subcategories_to_use = []  # Uses all if []

        if return_probabilities:
            return {
                "main_categories": main_categories_to_use,
                "sub_categories": subcategories_to_use,
                "main_probs": prediction["main_probs"],
                "sub_probs": prediction["sub_probs"]
                if "sub_probs" in prediction.keys()
                else {},
            }
        else:
            return {
                "main_categories": main_categories_to_use,
                "sub_categories": subcategories_to_use,
            }