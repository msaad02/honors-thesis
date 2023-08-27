"""
This uses the transformer model from scratch-model/transformer.ipynb to train a model, without
the excess comments and explanations. This is just the code, and is meant to be run in a .py file.

Specifically, what I'm trying to accomplish here is to get an end to end pipeline that can be
used to train a model, and then save that model to a file.
"""

# NOTE: Installing tensorflow_datasets and tensorflow_text updated tf to 2.13.0 from 12.2.1, 
# if it breaks revert and find correct versions

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # stop showing tensorflow logs...

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text

# From tf_dataset.py in scratch-model, this gets the tf data pipeline
from tf_dataset import get_datasets

train_ds, val_ds, text_processor = get_datasets()