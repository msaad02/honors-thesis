"""
This file contains the code to run inference on the scratch model.

To use it, import the ScratchModel class from this file and call it like so:

>>> from scratch_model.inference import ScratchModel
>>> model = ScratchModel()
>>> model("How can I apply to SUNY Brockport?")

Optionally, you can stream the output of the model by setting the stream
parameter in the call to True. This will return a generator object that
you can iterate over to get the output of the model one token at a time.

>>> from scratch_model.inference import ScratchModel
>>> from IPython.display import clear_output # If using ipynb
>>> model = ScratchModel()
>>> for token in model("How can I apply to SUNY Brockport?", stream=True):
...     print(token)
...     clear_output(wait=True) # If using ipynb
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # prevent tensorflow logs

# Set path to parent directory so we can import from other folders.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .dataset import standardize     # Crucial to reload the text processor
from .model import Transformer       # Model architecture
import tensorflow as tf
import json
import re

class ScratchModel(tf.Module):
    "Main class to run inference on the scratch model"
    def __init__(
        self,
        model_dir: str = "./models/transformer_v5/"
    ):
        # Load model parameters
        with open(f"{model_dir}params.json", "r") as f:
            config = json.load(f)

        # Load model
        self.transformer = Transformer(**config)
        _ = self.transformer.load_weights(model_dir).expect_partial()

        # Get text processor and vocab
        text_processor_model = tf.keras.models.load_model(model_dir + "text_processor")
        self.text_processor = text_processor_model.layers[0]

        self.vocab = self.text_processor.get_vocabulary()
        self.vocab_tf = tf.constant(self.vocab)

        # Load word mappings for capitalization
        self.word_list = {
            "brockport": "Brockport",
            "suny": "SUNY",
            "new york": "New York",
            "new york state": "New York State",
            "new york city": "New York City",
            "fafsa": "FAFSA",
            "rochester": "Rochester",
            "buffalo": "Buffalo",
            "computer science": "Computer Science",
            "math": "Math",
            "english": "English",
            "history": "History",
            "biology": "Biology",
            "chemistry": "Chemistry",
            "physics": "Physics",
            "psychology": "Psychology"
        }

    def _clean_string(self, text: str):
        "Cleans the string to be more readable using some simple rules"

        # Capitalize words that should be capitalized
        for word in self.word_list.keys():
            text = text.replace(word, self.word_list[word])
        
        text = text.removeprefix("[START] ")
        text = text.removesuffix(" [END]")

        # Remove spaces before punctuation marks
        text = re.sub(r'\s+([?.!,;])', r'\1', text)

        # Capitalize the first letter after punctuation marks and first letter of string
        text = re.sub(r'([?.!;]) (\w)', lambda x: x.group(1) + " " + x.group(2).upper(), text) 
        text = text[0].upper() + text[1:]
        return text

    # @tf.autograph.experimental.do_not_convert
    def _predict_next(self, question, output_array, i):
        "Predicts the next token given the question and the output array"
        output = tf.transpose(output_array.stack())
        prediction = self.transformer([question, output], training=False)
        prediction = prediction[:, -1:, :]
        prediction_id = tf.argmax(prediction, axis=-1)
        output_array = output_array.write(i+1, prediction_id[0])
        output = tf.transpose(output_array.stack())

        text = tf.strings.reduce_join(
            tf.map_fn(lambda x: self.vocab_tf[x], tf.squeeze(output), dtype=tf.string), separator=" "
        )

        return prediction_id, text, output_array
    
    def _stream_result(self, question, output_array, max_tokens, end):
        "Streams the result of the prediction"
        for i in tf.range(max_tokens):
                prediction_id, text, output_array = self._predict_next(question, output_array, i)
    
                if prediction_id == end:
                    break

                yield self._clean_string(text.numpy().decode("utf-8"))

    def _return_result(self, question, output_array, max_tokens, end):
        "Returns the result of the prediction"
        for i in tf.range(max_tokens):
                prediction_id, text, output_array = self._predict_next(question, output_array, i)
    
                if prediction_id == end:
                    break
        
        return self._clean_string(text.numpy().decode("utf-8"))

    def __call__(self, question: str, max_tokens: int = 256, stream: bool = False):
        "Oversees the prediction process. Returns a generator if stream=True"
        question = tf.convert_to_tensor([question])
        question = self.text_processor(question).to_tensor()

        start_end = self.text_processor([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        if stream:
            return self._stream_result(question, output_array, max_tokens, end)
        else:
            return self._return_result(question, output_array, max_tokens, end)