# Using the dataset I'm trying to use from model 
# finetuning and training from scratch, want to put the code here that I put it on HF with.

from datasets import load_dataset
import pickle
import re
from itertools import chain
import random
import json

# NOTE: May need to login to huggingfacehub if you want to push to hub
push_preformatted_dataset_to_hub = False     # Set true if you want to push the preformatted dataset to huggingface hub
push_formatted_dataset_to_hub = False          # Set true if you want to push the formatted dataset to huggingface hub

# Define names IF pushing to hub
preformatted_dataset_name = "msaad02/preformat-ss-cleaned-brockport-qa"     
formatted_dataset_name = "msaad02/formatted-ss-cleaned-brockport-qa"        

# JSON file name
json_file_name = "/home/msaad/workspace/honors-thesis/data_collection/data/cleaned_ss_dataset.json"

# ------------------------------ #
# Cleanup and format the dataset #
# ------------------------------ #

# Currenly this script uploads a small portion of dataset. This specific portion contains only the URLs included in version 1 of semantic search. 
# These include broad information about the school, and tend to be the most information rich webpages. (6% ish, see more details in data_cleaning_for_ss.py)

# Loading in relevant data.
gpt_output   = pickle.load(open('/home/msaad/workspace/honors-thesis/data_collection/data/gpt_output.p', 'rb'))
cleaned_dict = pickle.load(open('/home/msaad/workspace/honors-thesis/data_collection/data/cleaned_url_response_dict.p', 'rb'))

# Filter large dataset to small dataset
small_dataset = {k: v for k, v in gpt_output.items() if k in cleaned_dict.keys()}

# NOTE THIS FUNCTION IS TAKEN DIRECTLY FROM 'data-collection/data/gpt_data_generation.ipynb'
def generate_json(gpt_output: dict[str, str], filename: str = json_file_name) -> None:
    """
    Parses GPT output into a JSON file. This function uses regex to parse for all the instructions (questions), and 
    outputs (answers) from the GPT output which is a dictionary. The parsed data is returned into a Python list of 
    dictionaries, which is appended for each webpage. This list is shuffled to mix up the questions and then dumped 
    to a JSON file.

    Args:
        gpt_output (dict): The dictionary containing the GPT output.
        filename (str): The JSON filename to write the parsed and shuffled list of dictionaries to.
    """

    # The regular expression pattern for a JSON object with "instruction" and "output"
    pattern = r'"instruction":\s*"(.*?)",.*?"output":\s*"(.*?)"'

    def extract_data(s):
        matches = re.findall(pattern, s, flags=re.DOTALL)
        # Add a conditional filter in the list comprehension
        data = [{"instruction": m[0], "output": m[1]} for m in matches if m[0] and m[1] and '"' not in m[0] and '"' not in m[1]]
        return data

    jsonqa = []

    for value in gpt_output.values():
        clean_value = extract_data(value)
        jsonqa.append(clean_value)

    jsonqa = list(chain(*jsonqa))

    random.shuffle(jsonqa)

    # Write to a JSON file
    with open(filename, 'w') as f:
        json.dump(jsonqa, f, indent=4)  # Dump the entire list at once


print("Filtering to", len(small_dataset), "webpages (" + str(round(len(small_dataset)/ len(gpt_output) * 100, 2)) + "% of full dataset)")

# Generate the new dataset 
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