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

from datasets import load_dataset
import pickle
import re
from itertools import chain
import random
import json
import os

RAW_SCRAPER_OUTPUT = "data/raw_scraper_output.json"
GPT_3_5_DATA_QA = "data/gpt_3.5_data_qa.csv"
GPT_4_DATA_QA = "data/gpt_4_data_qa.csv"
CATEGORIZED_DATA = "data/categorized_data.json"

# ---------------------------------------------------------- #
# 1 - Upload raw scraper output

# To accomodate the huggingface dataset object, we need to convert the raw scraper output
# from a dictionary to a list of dictionaries
with open(RAW_SCRAPER_OUTPUT) as f:
    raw_scraper_output = json.load(f)
    
raw_scraper_output_hf = [{k: raw_scraper_output[k]} for k in raw_scraper_output.keys()]

with open("data/TEMP_raw_scraper_output_hf.json", "w") as f:
    json.dump(raw_scraper_output_hf, f)

# Upload to huggingface hub
dataset = load_dataset("json", data_files="data/TEMP_raw_scraper_output_hf.json")
dataset.push_to_hub("msaad02/raw-scraper-output")

# remove the hf file
os.remove("data/TEMP_raw_scraper_output_hf.json")

# ---------------------------------------------------------- #
# 2 - Upload GPT-3.5 data

dataset = load_dataset("csv", data_files=GPT_3_5_DATA_QA)
dataset.push_to_hub("msaad02/gpt-3.5-data-qa")

# ---------------------------------------------------------- #
# 3 - Upload GPT-4 data

dataset = load_dataset("csv", data_files=GPT_4_DATA_QA)
dataset.push_to_hub("msaad02/gpt-4-data-qa")

# ---------------------------------------------------------- #
# 4 - Upload categorized data




exit()

# NOTE: May need to login to huggingfacehub if you want to push to hub
push_preformatted_dataset_to_hub = False     # Set true if you want to push the preformatted dataset to huggingface hub
push_formatted_dataset_to_hub = False          # Set true if you want to push the formatted dataset to huggingface hub

# Define names IF pushing to hub
preformatted_dataset_name = "msaad02/preformat-ss-cleaned-brockport-qa"     
formatted_dataset_name = "msaad02/formatted-ss-cleaned-brockport-qa"        

# JSON file name
json_file_name = "/home/msaad/workspace/honors-thesis/data_collection/data/cleaned_ss_dataset.json"


generate_json(small_dataset)

# ---------------------------------------------------------- #
# Formatting the responses for the dataset for training step #
# ---------------------------------------------------------- #

# Load in the JSON dataset as a hugginface dataset object
dataset = load_dataset("json", data_files=json_file_name)


if push_preformatted_dataset_to_hub:
    dataset.push_to_hub(preformatted_dataset_name)

def format_prompt(example):
    """
    Format the prompt of the model. Expects huggingface dataset object input with 2 fields: instruction and output.
    """
    prompt = f"""Below is an inquiery related to SUNY Brockport - from academics, admissions, and faculty support to student life. Prioritize accuracy and brevity."

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

    return {'text': prompt}

# Map format_prompt on inputs
dataset = dataset.map(format_prompt, remove_columns=['instruction', 'output'])

if push_formatted_dataset_to_hub:
    dataset.push_to_hub(formatted_dataset_name)

# Sanity check
print(dataset['train']['text'][0])