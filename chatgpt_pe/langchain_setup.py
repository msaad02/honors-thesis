"""
This script creates a persistable vector database using Chroma and langchain
"""

import re
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader

data_path = "/home/msaad/workspace/honors-thesis/data-collection/data/cleaned_data.csv"
vectordb_persist_dir = "/home/msaad/workspace/honors-thesis/data-collection/data/chroma_persist_dir"

# Load in data
loader = CSVLoader(file_path=data_path, source_column='url')
data = loader.load()

# Chunk the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(data)

# Standardizing the content for each chunk
def standardize_string(input_string):
    # Step 1: Remove '\n' characters and replace them with periods
    standardized_string = input_string.replace('\n', ' ')

    # Step 2: Standardize the number of spaces
    standardized_string = re.sub(r'\s+', ' ', standardized_string)

    # Step 3: Remove non-alphanumeric characters at the start of the string
    standardized_string = re.sub(r'^[^a-zA-Z0-9]+', '', standardized_string)

    return standardized_string.strip()


# Create new 'texts' with some additional filters
texts_cleaned = []

# Iterate over texts page_content category with this cleaning method.
for id in range(len(texts)):
    texts[id].page_content = standardize_string(texts[id].page_content)

    if len(texts[id].page_content) > 100:
        texts_cleaned.append(texts[id])

texts_cleaned

embedding_function = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents = texts_cleaned,
    embedding = embedding_function,
    persist_directory = vectordb_persist_dir
)

vectordb.persist()