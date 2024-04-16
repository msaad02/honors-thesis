"""
For some reason, most likely due to how the model is implemented, the ScratchModel
runs EXCEPTIONALLY slow. Again, I'm not sure why exactly, but the scratch model is
a small transformer model that should not be running this slow. This script aims to
re-implement the  ScratchModel in PyTorch using their optimized Transformer module
and see if that fixes the issue.

All the parameters will be the same, so we can just copy them over. The only thing
that will change is the model implementation and data loading.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # prevent tensorflow logs

# Set path to parent directory so we can import from other folders.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
import torch.nn as nn
import torch
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Model ------------------------------------------------
EPOCHS = 13
BATCH_SIZE = 64
NUM_LAYERS = 6
D_MODEL = 512
DFF = 2048
NUM_HEADS = 8
DROPOUT_RATE = 0.1

model = nn.Transformer(
    d_model=D_MODEL,
    nhead=NUM_HEADS,
    num_encoder_layers=NUM_LAYERS,
    num_decoder_layers=NUM_LAYERS,
    dim_feedforward=DFF,
    dropout=DROPOUT_RATE,
    activation='relu'
).to(device)

# -------- Data ------------------------------------------------

dataset = load_dataset("msaad02/brockport-gpt-4-qa")['train'].to_pandas()
dataset.to_numpy()

