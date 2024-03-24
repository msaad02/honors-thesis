"""
Creating the dataset used for evaluation.

Remember, we use a combination of evaluation on the test set
and evaluation on the altered training data to get a full picture
of the model's performance. This script will create the altered
training data for evaluation, and combine it with the test set
to create the evaluation data.
"""

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import pandas as pd

# We us the full test set, but only a sample of the training set
TRAINING_SAMPLE_SIZE = 1000

tqdm.pandas()
client = OpenAI()

# Load data
dataset = load_dataset("msaad02/brockport-gpt-4-qa")

# Sample data
train_data = dataset['train'].to_pandas()[['question', 'answer']].sample(n=TRAINING_SAMPLE_SIZE)
train_data['type'] = 'train'

# Alter the training data
def adjust_question(question: str) -> str:
    "Slightly adjust the question."

    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "You are a helpful assistant who slightly modifies questions while retaining their meaning. Only make slight modifications such as changing to similar words, or rephrasing the question. Do not change the meaning of the question. Only respond with the question, do not preface it with anything. The question is: " + question}
        ],
        temperature=0.8,
        max_tokens=100
    )
    return result.choices[0].message.content

train_data['question'] = train_data['question'].progress_apply(adjust_question)


test_data = dataset['test'].to_pandas()[['question', 'answer']]
test_data['type'] = 'test'

# Combine the data
eval_data = pd.concat([train_data, test_data]).sample(frac=1)

# Save the data
eval_data.to_csv('./data/evaluation_data.csv', index=False)


