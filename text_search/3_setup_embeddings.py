"""
This scripts creates the embeddings for all the data.

It is mainly split into two parts:
1. Prepare the raw data for embedding (chunking, cleaning, etc.)
2. Create the embeddings for each category and subcategory

Importantly, in the second part, we are storing everything in a specific way
so that we can easily access the embeddings for each category and subcategory,
as well as the underlying data that was used to create the embeddings.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pandas as pd
import pickle

# Create all the embeddings for the categories and subcategories
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# ---- PART 1 ---------------------------------------------------------------
# Prepare the raw data for embedding (chunking, cleaning, etc.)

# load in the data
categorized_data = load_dataset("msaad02/categorized-data", split="train").to_pandas()
categorized_data = categorized_data.dropna(subset=['category']).reset_index(drop=True)

def filter_out_small_strings(input_string):
    "There are many 'sentences' that are really just titles for sections. These don't add much value to the embeddings so we'll filter them out."
    split_string = pd.Series(input_string.split("\n"))  # Split by newlines (which is how the data is formatted)
    tf_mask = [len(i) > 50 for i in split_string]       # Filter out groups that are less than 50 characters long
    out = split_string[tf_mask].str.strip()             # Apply filter and strip whitespace
    out = out.str.removeprefix("- ").str.removesuffix(".").to_list()    # Remove leading bullets and trailing periods
    return ". ".join(out)       # Join the list back into a string and add periods back in

cleaned = categorized_data['data'].apply(filter_out_small_strings)
cleaned = cleaned[cleaned.str.split(" ").str.len() > 100] # filter out data with less than 100 words

## NOTE: Finds the average length of sentences in the dataset.
## We are aiming to have 2-3 sentences per chunk, so we need to found a character count to use. 
## This tells us that. Our output was ~346 characters.
# cleaned.apply(lambda x: pd.Series([len(sentences) for sentences in x.split('. ')]).mean()).mean() * 2.5

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=25,
    separators=[".", "?", "!"]
)

raw_chunks = cleaned.apply(text_splitter.split_text)

def clean_chunks(chunks):
    "RecursiveCharacterTextSplitter has weird behavior... It puts the punctuation from previous chunks into the next chunk. This cleans that up."
    for i in range(len(chunks)-1):
        chunks[i] += chunks[i+1][:1]
        chunks[i+1] = chunks[i+1][2:]
    return chunks

chunked_data = pd.Series(raw_chunks.copy()).apply(clean_chunks)
categorized_data['chunked_data'] = chunked_data
categorized_data = categorized_data.dropna().reset_index(drop=True)
categorized_data = categorized_data.explode('chunked_data').loc[:, ['category', 'subcategory', 'chunked_data']].reset_index(drop=True)

# Filter to only chunks with less than 100 words
# Inspecting the data, mostly chunks with >100 words are just a bunch of bullet points of long lists about profs, programs, etc.
categorized_data = categorized_data[categorized_data['chunked_data'].str.split().str.len() < 100].reset_index(drop=True)

# # This is the distribution of the number of words per chunk. Should look nice and pretty (~Normal).
# categorized_data['chunked_data'].str.split().str.len().hist(bins=30)


# ---- PART 2 ---------------------------------------------------------------
# Create embeddings for each category and subcategory

df = categorized_data.rename(columns={'chunked_data': 'data'})

# Line of code from `train_subcat_classifier.py`
categories_with_subcategories = df['category'].unique()[df.groupby(["category"])['subcategory'].nunique() > 4]

# Create embeddings for all the categories and subcategories and store them in a dictionary
embeddings = {}
data = {}

for category in df['category'].unique():
    print("Writing embeddings for category: ", category, "...")
    category_df = df[df['category'] == category]

    if category in categories_with_subcategories:
        # first get the embeddings for the main category
        data_to_embed = category_df[category_df['subcategory'].isnull()]['data'].to_list()
        if data_to_embed != []:
            embeddings[category] = model.encode(data_to_embed, normalize_embeddings=True)
            data[category] = data_to_embed

        # split the data into subcategories
        subcategories = category_df[category_df['subcategory'].notnull()]['subcategory'].unique()
        for subcategory in subcategories:
            data_to_embed = category_df[category_df['subcategory'] == subcategory]['data'].to_list()
            if data_to_embed != []:
                embeddings[f"{category}-{subcategory}"] = model.encode(data_to_embed, normalize_embeddings=True)
                data[f"{category}-{subcategory}"] = data_to_embed
    else:
        data_to_embed = category_df['data'].to_list()
        if data_to_embed != []:
            embeddings[category] = model.encode(data_to_embed, normalize_embeddings=True)
            data[category] = data_to_embed

data = {'embeddings': embeddings, 'data': data}

# Save the embeddings
with open("embeddings.pickle", "wb") as f:
    pickle.dump(data, f)

print("\nComplete!")