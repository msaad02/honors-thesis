"""
This script serves as a data cleaning and preparation step in a larger pipeline designed to populate a vector database using 
content scraped from the Brockport website. This process is essential in creating a structured and clean dataset from unstructured
web data, enabling more efficient querying and analysis further down the line.

The data that this script processes has been previously collected using a web scraping method. The scraper has navigated through 
the Brockport website and stored the HTML of each page in a Python dictionary, with the URLs serving as keys and the HTML content 
as values. This raw data is stored in the form of pickle files in the data folder.

The primary task of this script is to parse this HTML content, identify meaningful sentences or blocks of text, and filter out 
unwanted characters or strings. This is done using BeautifulSoup to parse the HTML and regular expressions to apply various text 
filters.

More specifically, the get_text() function within the script parses the HTML content, breaks it down into individual strings, 
and applies a series of conditions to filter the content. These conditions include:
1. Identifying long blocks of text (considered here as having 15 or more words)
2. Filtering out any strings that contain characters other than alphanumeric, certain punctuation marks (.,/?!:;$#&+*()")
3. Removing any strings that contain more than three consecutive whitespace characters

The result of that description is a list of cleaned html pages which are then written to a CSV file. That csv file will serve as an 
accessible input to a vector database, which is more specifically addressed later in this script.

Once the csv is generated, it is then split line by line into txt files. The tool of choice for this project, langchain, opts for
data folders to be specified instead of large files. Then, any txt file inside that folder can be read, have embeddings
generated, etc. etc torwards the end of having a functional vector database. So, obviously I need to split up that csv file. 
That is done in the last part of this script, which reads in the csv, and iterates over each line, creating a txt file
for each one. This should facilite efficient storage, retrieval, and analysis of the Brockport website content.
"""

from bs4 import BeautifulSoup
import pickle
import re
import requests
from tqdm import tqdm
import csv
import pandas as pd

"""
NOTE July 15th Update: 
Some of the naming here is slightly off now. Previously where 'sentences' were used, I am now generating 'chunks'.
A chunk is basically just a group of all the sentences inside the response. In the first implementation, with sentences,
when it came time to use semantic search the results were wayy too short and off the mark in terms of context. It 
turns out, that most of the sentences are meaningfully related, so to arbitrarily cutoff at every newline doesn't
make sense - hence now I'm joining the sentences together for each of the responses.

Basically, think chunk as the text portion of a webpage. If you have 1000 text rich html responses, you should end up with a 1000 row csv.
"""

data_folder = "/home/msaad/workspace/honors-thesis/data-collection/data/"
responses_dict = pickle.load(open(data_folder + "scraper_output.p", "rb"))

def get_text(response: requests.models.Response) -> list:
    """
    Parses HTML for 'sentences', as described above.
    """
    # Your BeautifulSoup object
    soup = BeautifulSoup(response.text, 'html.parser')

    # Create an empty list to hold the sentences
    long_texts = []

    # Define custom punctuation
    custom_punctuation = ',./?!:;$#&+*()"'

    # Loop through all the strings in the BeautifulSoup object
    for string in soup.stripped_strings:
        # Consider a long text as a text having 15 or more words
        if len(string.split()) >= 15:
            # Check if the string contains only alphanumeric, whitespace and custom punctuation characters
            if all(c.isalnum() or c.isspace() or c in custom_punctuation for c in string):
                # Check if the string contains more than 3 whitespaces in a row
                if not re.search(' {3,}', string):
                    long_texts.append(string)

    # NEW: Added so semantic search queries return bigger, more in context results.
    # "chunk" is a string of all the long sentences (as described above) separated by a newline.
    chunk = " ".join(long_texts)
    
    # OLD: return long_texts
    return chunk

print("Begin fetching all the sentences...")
# Wrap iterable with tqdm() to see progress bar
# Maps get_text over the response dictionary to generate a list of chunks. Filers out the empty ones, and saves to `all_sentences`
all_sentences = list(filter(lambda x: x != '', map(get_text, tqdm(responses_dict.values()))))

# OLD: 
# all_sentences = [sentence for response in tqdm(responses_dict.values()) for sentence in get_text(response)]
print(f"Saving off {len(all_sentences)} sentences.")

# Save off to a csv file
csv_name = "chunks_from_html.csv"
with open(data_folder + csv_name, 'w', newline='') as f:
    writer = csv.writer(f)
    for sentence in all_sentences:
        writer.writerow([sentence])


# Now to split up the file into chunks for langchain
chunks_csv = pd.read_csv(data_folder + csv_name)

for row_index in range(len(chunks_csv)):
    # open the file with write mode
    with open(f"{data_folder}vectordb_filestore/chunk_{row_index}.txt", 'w') as file:
        # write a row of the csv to the file
        file.write(chunks_csv.iloc[row_index, 0])