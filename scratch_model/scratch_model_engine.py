"""
Simple script to create class to run scratch model.

Expects tensorflow model that when ran, outputs a string.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # prevent tensorflow logs

import tensorflow as tf
import re

tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True
) # stop tensorflow from taking up all GPU memory

class ScratchModelEngine():
    """
    Simple class to interact with scratch model.
    """
    def __init__(
            self,
            model_path: str = "/home/msaad/workspace/honors-thesis/scratch_model/models/translator",
        ):
        """
        Initialize the scratch model.
        """
        self.model = tf.saved_model.load(model_path)

    def _clean_string(self, s):
        # Remove [START] and [END] tags and trim whitespaces
        s = s.replace("[START]", "").replace("[END]", "").strip()
        
        # Remove spaces before punctuation like ",", ".", "!", etc.
        s = re.sub(r'\s+([?.!,;])', r'\1', s)
        
        # Capitalize the first letter after punctuation marks
        s = re.sub(r'([?.!;]) (\w)', lambda x: x.group(1) + " " + x.group(2).upper(), s)

        s = s[0].upper() + s[1:] # Capitalize the first letter of the string        
        return s
    
    def _auto_capitalize(self, s, word_list):
        word_list_lower = set(map(str.lower, word_list))  # Convert to lowercase set for faster lookup
        words = re.findall(r'\b\w+\b', s)  # Extract words, ignoring punctuation
        for _, word in enumerate(words):
            if word.lower() in word_list_lower:
                s = re.sub(r'\b' + re.escape(word) + r'\b', word.capitalize(), s)
        return s

    def __call__(
            self, 
            question: str,
        ):
        """
        Run the scratch model on the question to get an answer.
        """
        answer = self.model(question).numpy().decode('utf-8').strip()

        capitalize_list = ['Brockport', 'suny']

        answer = self._clean_string(answer)
        answer = self._auto_capitalize(answer, capitalize_list)

        return answer