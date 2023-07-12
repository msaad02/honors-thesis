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

The result of this script is a list of cleaned sentences which are then written to a CSV file. This file will serve as an 
accessible input to a vector database, facilitating efficient storage, retrieval, and analysis of the Brockport website content.
"""
from bs4 import BeautifulSoup
import pickle
import re
import requests
from tqdm import tqdm
import csv

data_folder = "/home/msaad/workspace/honors-thesis/data-collection/data/"
responses_html = pickle.load(open(data_folder + "scraper_output.p", "rb"))

def get_text(response: requests.models.Response) -> list:
    """
    Parses HTML for 'sentences', as described above.
    """
    # Your BeautifulSoup object
    soup = BeautifulSoup(response.text, 'html.parser')

    # Create an empty list to hold the sentences
    long_texts = []

    # Define your custom punctuation
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

    # Now, long_texts list contains all the sentences fitting the criteria described above.
    return long_texts

print("Begin fetching all the sentences...")
# Wrap your iterable with tqdm() to see progress bar
all_sentences = [sentence for response in tqdm(responses_html.values()) for sentence in get_text(response)]
print(f"Saving off {len(all_sentences)} sentences.")

# Save off to a csv file
with open(data_folder + "sentences.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    for sentence in all_sentences:
        writer.writerow([sentence])
