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

from typing import List, Tuple
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

model.train()

# -------- Data ------------------------------------------------

dataset = load_dataset("msaad02/brockport-gpt-4-qa")['train'].to_pandas()

class Seq2SeqDataset(Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context = torch.tensor([self.vocab[token] for token in self.tokenizer(context)])
        target = torch.tensor([self.vocab[token] for token in self.tokenizer(target)])
        return context, target

def build_vocab(data, tokenizer):
    token_generator = (token for _, sent in data for token in tokenizer(sent))
    vocab = build_vocab_from_iterator(
        iterator = [token_generator], 
        specials=["<unk>", "<pad>", "<sos>", "<eos>"], 
        special_first=True, 
        min_freq=5
    )
    return vocab

def collate_batch(batch):
    contexts, targets = zip(*batch)
    pad_idx = vocab['<pad>']
    contexts = pad_sequence(contexts, padding_value=pad_idx, batch_first=True)
    targets = pad_sequence(targets, padding_value=pad_idx, batch_first=True)
    return contexts, targets

# Create a list of tuples (context, target)
data = list(zip(dataset['question'].tolist(), dataset['answer'].tolist()))
np.random.shuffle(data)

# Randomly split data into train and validation
split_idx = int(0.85 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

# Define tokenizer and vocabulary
tokenizer = get_tokenizer('basic_english')

vocab = build_vocab(train_data + val_data, tokenizer)
vocab.set_default_index(vocab["<unk>"])

# Create datasets
train_dataset = Seq2SeqDataset(train_data, vocab, tokenizer)
val_dataset = Seq2SeqDataset(val_data, vocab, tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


# -------- Training ------------------------------------------------
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

pad_idx = vocab['<pad>']
sos_idx = vocab['<sos>']
eos_idx = vocab['<eos>']

# Define the loss function and optimizer
loss_fn = CrossEntropyLoss(ignore_index=pad_idx)
optimizer = Adam(model.parameters(), lr=0.0001)

# Define the training function
def train(model, data_loader, optimizer, loss_fn, device):
    total_loss = 0

    for contexts, targets, context_lens, target_lens in data_loader:

        print(f"Contexts: {contexts}")
        print(f"Targets: {targets}")
        print(f"Context lens: {context_lens}")
        print(f"Target lens: {target_lens}")



        # Move tensors to the right device
        contexts = contexts.to(device)
        targets = targets.to(device)

        # Forward pass
        output = model(contexts, targets) # The decoder input should be all tokens except the last.

        # Compute the loss
        loss = loss_fn(output.reshape(-1, output.size(-1)), targets[1:, :].reshape(-1)) # Shifted targets for computing loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = train(model, train_loader, optimizer, loss_fn, device)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
