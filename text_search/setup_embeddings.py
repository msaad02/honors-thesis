from sentence_transformers import SentenceTransformer
import pickle
import pandas as pd

# Create all the embeddings for the categories and subcategories
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Load the categorized data
df = pd.read_csv("../data_collection/data/categorized_data.csv").dropna()

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