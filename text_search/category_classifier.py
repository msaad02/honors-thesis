"""
This module contains a question classifier that predicts the category of a given question. The model is trained inside of `train_question_classifier.py`. The predicted category can be obtained by calling the `predict` method of the `QuestionClassifier` class.

Example usage:
    classifier = QuestionClassifier(model_dir='model')
    category = classifier.predict("How can I get financial aid?", return_probabilities=True)
    print(category)
"""

import torch
import torch.nn as nn
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

class QuestionClassifier:
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
