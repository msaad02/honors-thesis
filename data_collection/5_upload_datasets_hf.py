"""
Uploads all the datasets we created to huggingface hub. This includes:

1. "raw_scraper_output.json": The raw output from the scraper. Output of 1_scraping.py
2. "gpt_3.5_data_qa.csv": The data generated from GPT-3.5. Output of 3_qa_generation.py
3. "gpt_4_data_qa.csv": The data generated from GPT-4. Output of 3_qa_generation.py
4. "categorized_data.json": The categorized data. Output of 4_categorize_data.py

NOTE: The output of 2_cleaning.py is included in gpt_3.5_data_qa.csv and gpt_4_data_qa.csv.

Finally, this script also creates and uploads a dataset containing separate, full lists 
of questions and answers from the GPT-3.5 and GPT-4 generated datasets. These are named:

1. "brockport_gpt_3.5_qa.json"
2. "brockport_gpt_4_qa.json"
"""

# NOTE: May need to login to huggingfacehub if you want to push to hub

# Ideally you do this with `huggingface-cli login`, but it stopped working for
# me so I just manually set it with an environment variable.

from datasets import load_dataset, Dataset
import pandas as pd
import json
import os

RAW_SCRAPER_OUTPUT = "data/raw_scraper_output.json"
GPT_3_5_DATA_QA = "data/gpt_3.5_data_qa.csv"
GPT_4_DATA_QA = "data/gpt_4_data_qa.csv"
CATEGORIZED_DATA = "data/categorized_data.csv"

# ---------------------------------------------------------- #
# 1 - Upload raw scraper output

# To accomodate the huggingface dataset library we need to convert the raw scraper output
with open(RAW_SCRAPER_OUTPUT) as f:
    raw_scraper_output = json.load(f)

raw_scraper_output_df = pd.DataFrame({
    'url': [k for k in raw_scraper_output.keys()],
    'data': [raw_scraper_output[k] for k in raw_scraper_output.keys()]
})

dataset = Dataset.from_pandas(raw_scraper_output_df)
dataset.push_to_hub("msaad02/raw-scraper-output", token=os.getenv("HUGGINGFACE_TOKEN"))

# ---------------------------------------------------------- #
# 2 - Upload GPT-3.5 data

dataset = load_dataset("csv", data_files=GPT_3_5_DATA_QA)
dataset.push_to_hub("msaad02/gpt-3.5-data-qa", token=os.getenv("HUGGINGFACE_TOKEN"))

# ---------------------------------------------------------- #
# 3 - Upload GPT-4 data

dataset = load_dataset("csv", data_files=GPT_4_DATA_QA)
dataset.push_to_hub("msaad02/gpt-4-data-qa", token=os.getenv("HUGGINGFACE_TOKEN"))

# ---------------------------------------------------------- #
# 4 - Upload categorized data

dataset = load_dataset("csv", data_files=CATEGORIZED_DATA)
dataset.push_to_hub("msaad02/categorized-data", token=os.getenv("HUGGINGFACE_TOKEN"))

# ---------------------------------------------------------- #
# 5 - Upload full lists of questions and answers

def get_qa_dataset(df: pd.DataFrame) -> Dataset:
    """
    Creates a huggingface dataset from the outfiles of 3_qa_generation.py.

    Note that this ignores null API responses.
    """
    questions = []
    for _, row in df.loc[df['questions'].notnull(), ['url', 'questions']].iterrows():
        url, question = row['url'], row['questions']

        list_of_questions = json.loads(question)
        list_of_questions = [{
            'url': url, 
            'question': q['question'], 
            'answer': q['answer']
        } for q in list_of_questions]

        questions.extend(list_of_questions)

    dataset = Dataset.from_list(questions)
    # For consistency in comparison evals train/test split set here
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset

# GPT-3.5
gpt35_dataset = get_qa_dataset(pd.read_csv(GPT_3_5_DATA_QA))
gpt35_dataset.push_to_hub("msaad02/brockport-gpt-3.5-qa", token=os.getenv("HUGGINGFACE_TOKEN"))

# GPT-4
gpt4_dataset = get_qa_dataset(pd.read_csv(GPT_4_DATA_QA))
gpt4_dataset.push_to_hub("msaad02/brockport-gpt-4-qa", token=os.getenv("HUGGINGFACE_TOKEN"))

# ---------------------------------------------------------- #

print("All datasets uploaded to huggingface!")