"""
This script creates the Typesense schema and loads data into it. Currently, it
is setup to support OpenAI embeddings with 'text-embedding-ada-002', but this can
be changed to other embeddings (see Typesense documentation for more information).

An intermediate step here is taking our chunked data from `3_prep_for_embeddings.py`
and putting it into jsonl files that Typesense can read. This is done in the
`create_jsonl_files` function, and is promptly deleted after the data is loaded.


# Typesense Text Search

This is a simple implementation of typesense. It includes a script to index the data
and a class designed to search the data given a question. We have an implementation
that includes the question classifier as well.

Typesense, since I chose not to use the cloud version, is a self-hosted search engine.
It is pretty easy to use, but does require some setup. I would recommend using the
docker installation, since it is the easiest to get up and running. Here is their
official guide to getting started: https://typesense.org/docs/guide/install-typesense.html

After installation, run the following command to start typesense. Just be sure to change
the directory you installed to, api key, or port as needed.

```bash
docker run -p 8108:8108 \
            -v/home/msaad/typesense-data:/data typesense/typesense:0.25.2 \
            --data-dir /data \
            --api-key=xyz \
            --enable-cors
```
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
    {"name": "main_category", "type": "string"},
    {"name": "sub_category", "type": "string"},
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

# ------ Most of this is from `3_prep_for_embeddings.py`, with slight modifications ------
try: 
    categorized_data = pd.read_csv("../data_collection/data/chunked_data2.csv")
except:
    categorized_data = pd.read_csv("chunked_data2.csv")

# Remove url category, rename, and drop duplicates
df = categorized_data.rename({'category': 'main_category', 'subcategory': 'sub_category', 'chunked_data': 'context'}, axis=1)
df = df.drop_duplicates(subset=['context']).reset_index(drop=True)

df.to_csv("documents.csv", index=False)

# ------ End of `3_prep_for_embeddings.py` ------

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