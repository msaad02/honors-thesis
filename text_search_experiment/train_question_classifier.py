"""
This script trains a question classifier model using a dataset of categorized questions. It uses a CountVectorizer to convert the text data into numerical features and a neural network model for classification. The trained model is saved along with other necessary files for inference.

Keep in mind that this data is *heavily* unbalanced. To account for this, we are oversampling the data. As opposed to weighting the loss function, this method performs better, but regardless is not ideal.

The output of this file is a directory called "model" that contains the following files:
- category_classifier_model.pth: the trained model
- vectorizer.joblib: the CountVectorizer used to convert text to numerical features
- class_mappings.json: a dictionary mapping class names to integers and vice versa
- hyperparameters.json: a dictionary containing the hyperparameters used to train the model

The model can be used for inference by using the `QuestionClassifier` class from question_classifier.py
"""

from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
import torch.nn as nn
import pandas as pd
import torch
import joblib
import json
import os

# All useful parameters are here:
SAMPLE_SIZE = 2000
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.0001
HIDDEN_DIM = 256
DROPOUT = 0.5
SAVE_DIR = "model/"

# check SAVE_DIR exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Set default device based on whether CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("msaad02/categorized-data", split="train")
category_df = dataset.to_pandas()

qa_df = pd.concat([
    load_dataset("msaad02/brockport-gpt-4-qa", split="train").to_pandas(),
    load_dataset("msaad02/brockport-gpt-3.5-qa", split="train").to_pandas()
])

df = pd.merge(qa_df, category_df[["url", "category", "subcategory"]], on="url", how="left")

train_df = df.groupby("category").sample(n=SAMPLE_SIZE, replace=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)

vectorizer = CountVectorizer(lowercase=True, min_df=5)

train_vect = vectorizer.fit_transform(train_df['question'])

text_to_int = {text: idx for idx, text in enumerate(train_df['category'].unique())}
int_to_text = {idx: text for text, idx in text_to_int.items()}

train_df['category_vect'] = train_df['category'].apply(lambda x: text_to_int[x])

train_proportion = 0.8
validation_prop = 0.1
train_size = int(train_proportion * train_vect.shape[0])
validation_size = int(validation_prop * train_vect.shape[0])
end_validation_size = train_size + validation_size

X_train = torch.tensor(train_vect[:train_size].toarray(), dtype=torch.float)
y_train = torch.tensor(train_df['category_vect'][:train_size].to_numpy(), dtype=torch.long)
X_val = torch.tensor(train_vect[train_size:end_validation_size].toarray(), dtype=torch.float)
y_val = torch.tensor(train_df['category_vect'][train_size:end_validation_size].to_numpy(), dtype=torch.long)
X_test = torch.tensor(train_vect[end_validation_size:].toarray(), dtype=torch.float)
y_test = torch.tensor(train_df['category_vect'][end_validation_size:].to_numpy(), dtype=torch.long)

# Create DataLoaders for training and validation
train_loader = DataLoader(dataset=TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

# Model architecture
class CategoryClassifier(nn.Module):
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

# Instantiate the model
vocab_size = train_vect.shape[1]
hidden_dim = HIDDEN_DIM
output_dim = len(text_to_int)
dropout = DROPOUT

model = CategoryClassifier(vocab_size, hidden_dim, output_dim, dropout)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Move model to device
model = model.to(device)

# Training loop
train_losses = []
val_losses = []
for epoch in range(EPOCHS):
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    model.eval()
    for batch_idx, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# File to write the output
output_file = f'{SAVE_DIR}model_evaluation.txt'

# Evaluate the model on the test data
with torch.no_grad():
    model.eval()
    outputs = model(X_test.to(device))
    _, predicted = torch.max(outputs.data, 1)

# Calculate accuracy
accuracy = accuracy_score(y_test.cpu(), predicted.cpu())

# Calculate confusion matrix
confusion_mat = confusion_matrix(y_test.cpu(), predicted.cpu())

# Calculate precision, recall, and F1-score
classification_rep = classification_report(y_test.cpu(), predicted.cpu())

# Write the results to the file
with open(output_file, 'w') as f:
    f.write("Model Evaluation Results on Test Data:\n\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(confusion_mat, separator=', ') + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_rep + "\n")

print(f"Model evaluation results written to {output_file}")

# Save the model
torch.save(model.state_dict(), f'{SAVE_DIR}category_classifier_model.pth')

# save vectorizer
joblib.dump(vectorizer, f'{SAVE_DIR}vectorizer.joblib')

# save class mappings
with open(f'{SAVE_DIR}class_mappings.json', 'w') as f:
    json.dump({'text_to_int': text_to_int, 'int_to_text': int_to_text}, f)

# save hyperparameters
# Note that these are not all the hyperparameters, just the ones that are needed for inference
with open(f'{SAVE_DIR}hyperparameters.json', 'w') as f:
    json.dump({'vocab_size': vocab_size, 'hidden_dim': hidden_dim, 'output_dim': output_dim, 'dropout': dropout}, f)

print(f"Model saved to {SAVE_DIR} directory")