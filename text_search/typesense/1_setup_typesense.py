"""
This script creates the Typesense schema and loads data into it. Currently, it
is setup to support OpenAI embeddings with 'text-embedding-ada-002', but this can
be changed to other embeddings (see Typesense documentation for more information).

An intermediate step here is taking our chunked data from `3_prep_for_embeddings.py`
and putting it into jsonl files that Typesense can read. This is done in the
`create_jsonl_files` function, and is promptly deleted after the data is loaded.
"""

import typesense
import subprocess
import os
import sys
import pandas as pd

typesense_host: str = "localhost"
typesense_port: str = "8108"
typesense_protocol: str = "http"
typesense_collection_name: str = "brockport_data_v1"
typesense_api_key: str = "xyz"
openai_api_key: str = os.getenv("OPENAI_API_KEY")

client = typesense.Client({
    'nodes': [{
        'host': typesense_host,
        'port': typesense_port,
        'protocol': typesense_protocol
    }],
    'api_key': typesense_api_key,
    'connection_timeout_seconds': 60,
    'retry_interval_seconds': 5
})

# Define index schema
index_name = typesense_collection_name
schema = {
  "name": index_name,
  "fields": [
    {"name": "url", "type": "string"},
    {"name": "context", "type": "string"},
    {
      "name" : "embedding",
      "type" : "float[]",
      "embed": {
        "from": ["context"],
        "model_config": {
            "model_name": "openai/text-embedding-ada-002",
            "api_key": openai_api_key
        }
      }
    }
  ]
}

# Delete existing collection if it exists - but ask first!
try:
    client.collections[index_name].retrieve()

    input = input("Collection already exists! Enter 'y' to delete and recreate, or 'n' to exit >>> ")
    if input == 'y':
        client.collections[index_name].delete()
    else:
        print("Exiting...")
        exit()
except:
    pass

# Create collection
collection = client.collections.create(schema)

# Attempt to load in chunked data
try:
    chunked_data = pd.read_csv('../data_collection/data/chunked_data.csv')
except:
    chunked_data = pd.read_csv('chunked_data.csv')

if chunked_data is None or chunked_data.empty:
    print("Chunked data does not exist at path. Exiting.")
    sys.exit(1)

chunked_data.rename(columns={'chunked_data': 'context'}, inplace=True)
chunked_data.loc[:, ['url', 'context']].to_csv('documents.csv', index=False)

# Create jsonl file
if not subprocess.run(f'mlr --icsv --ojsonl cat documents.csv > documents.jsonl', check=True, shell=True):
    print("Failed to create jsonl file. Exiting.")
    print("Make sure you have Miller installed, see documentation and github https://typesense.org/docs/0.18.0/api/documents.html#import-a-csv-file and https://github.com/johnkerl/miller")
    sys.exit(1)

# Check if documents.jsonl exists
if not os.path.exists('documents.jsonl'):
    print("documents.jsonl does not exist. Exiting.")
    sys.exit(1)

print("jsonl file successfully created")

# Read and index JSONL file in batches
print("Indexing documents...")
jsonl_file_path = 'documents.jsonl'
with open(jsonl_file_path) as jsonl_file:
  client.collections[index_name].documents.import_(jsonl_file.read().encode('utf-8'), {'action': 'create'})

# Delete files
os.remove(jsonl_file_path)
os.remove('documents.csv')

# Done!
print("\n\nDone!")