"""
This script creates a tensorflow dataset for the brockport-gpt-4-qa dataset to feed into the model.

This also does some preprocessing of the data, such as adding start and end tokens, and standardizing the text.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # stop showing tensorflow logs

from datasets import load_dataset
import tensorflow as tf
import numpy as np

# Adding decorator to be able to serialize the function for saving the model
@tf.keras.utils.register_keras_serializable("Custom", name="text_standardization")
def standardize(text):
    "Text standardization function. Tries to make things uniform."
    text = tf.strings.lower(text) # Lowercase everything
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '') # Keep space, a to z and punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ') # Add spaces around punctuation.
    text = tf.strings.strip(text) # Strip whitespace.

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ') # Add start and end token
    return text


def get_datasets(batch_size: int = 64, max_vocab_size: int = 5000):
    """
    Loads in the data and returns a tensorflow dataset for training and validation.

    This also oversees the preprocessing of the data, such as start and end tokens,
    as well as standardizing the text. It returns the 
    """
    dataset = load_dataset("msaad02/brockport-gpt-4-qa")
    dataset = dataset['train'].to_pandas()

    context_raw = dataset['question'].to_list()
    target_raw = dataset['answer'].to_list()

    is_train_mask = np.random.uniform(size=(len(target_raw),)) < 0.8

    train_context = np.array(context_raw)[is_train_mask]
    train_target = np.array(target_raw)[is_train_mask]

    val_context = np.array(context_raw)[~is_train_mask]
    val_target = np.array(target_raw)[~is_train_mask]

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((train_context, train_target))
        .shuffle(len(context_raw))
        .batch(batch_size)
    )
    val_raw = (
        tf.data.Dataset
        .from_tensor_slices((val_context, val_target))
        .shuffle(len(context_raw))
        .batch(batch_size)
    )

    text_processor = tf.keras.layers.TextVectorization(
        standardize=standardize,
        max_tokens=max_vocab_size,
        ragged=True
    )

    text_processor.adapt(train_raw.map(lambda context, target: context))
    text_processor.adapt(train_raw.map(lambda context, target: target))

    def process_text(context, target):
        context = text_processor(context).to_tensor()
        target  = text_processor(target)
        targ_in = target[:,:-1].to_tensor()
        targ_out = target[:,1:].to_tensor()
        return (context, targ_in), targ_out

    train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
    val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

    return train_ds, val_ds, text_processor
