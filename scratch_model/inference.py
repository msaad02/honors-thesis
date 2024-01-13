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
>>> model = ScratchModel()
>>> for token in model("How can I apply to SUNY Brockport?", stream=True):
...     print(token)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # hide tensorflow logs

from dataset import standardize     # Crucial to reload the text processor
from model import Transformer       # Model architecture
import tensorflow as tf
import json
import re

class ScratchModel(tf.Module):
    "Main class to run inference on the scratch model"
    def __init__(
        self,
        model_dir: str = "./models/transformer_v4/",
        fix_capitalization_dir: str = "./fix_capitalization.json"
    ):
        # Load model parameters
        with open(f"{model_dir}/params.json", "r") as f:
            config = json.load(f)
        batch_size = config.pop("batch_size")

        # Load model
        self.transformer = Transformer(**config)
        _ = self.transformer.load_weights(model_dir)

        # Get text processor and vocab
        loaded_text_processor = tf.keras.models.load_model(model_dir + "text_processor")
        self.text_processor = loaded_text_processor.layers[0]

        self.vocab = self.text_processor.get_vocabulary()
        self.vocab_tf = tf.constant(self.vocab)

        # Load word list for capitalization
        with open(fix_capitalization_dir, "r") as f:
            self.word_list = json.load(f)

    def _clean_string(self, s: str):
        "Cleans the string to be more readable using some simple rules"
        # Automatically capitalize words that should be capitalized
        for word in self.word_list.keys():
            s = s.replace("word", self.word_list[word])
        
        s = s.replace("[START]", "").replace("[END]", "").strip() # Remove [START] and [END] tags
        s = re.sub(r'\s+([?.!,;])', r'\1', s) # Remove spaces before punctuation like ",", ".", etc.

        # Capitalize the first letter after punctuation marks and firstl letter of string
        s = re.sub(r'([?.!;]) (\w)', lambda x: x.group(1) + " " + x.group(2).upper(), s) 
        s = s[0].upper() + s[1:]
        return s

    @tf.autograph.experimental.do_not_convert
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

                yield self._clean_string(str(text))

    def _return_result(self, question, output_array, max_tokens, end):
        "Returns the result of the prediction"
        for i in tf.range(max_tokens):
                prediction_id, text, output_array = self._predict_next(question, output_array, i)
    
                if prediction_id == end:
                    break
    
        return self._clean_string(str(text))

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