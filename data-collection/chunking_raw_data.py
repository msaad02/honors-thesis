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
import os
import trafilatura

"""
NOTE July 15th Update: 
Some of the naming here is slightly off now. Previously where 'sentences' were used, I am now generating 'chunks'.
A chunk is basically just a group of all the sentences inside the response. In the first implementation, with sentences,
when it came time to use semantic search the results were wayy too short and off the mark in terms of context. It 
turns out, that most of the sentences are meaningfully related, so to arbitrarily cutoff at every newline doesn't
make sense - hence now I'm joining the sentences together for each of the responses.

Basically, think chunk as the text portion of a webpage. If you have 1000 text rich html responses, you should end up with a 1000 row csv.
"""

"""
NOTE July 16th Update:
It appears my data is actually quite imperfect. Atleast from the chunk_from_html.csv it is apparent that there are a lot of duplicates! 
Some of the webpages I will need to filter out, so that semantic search doesn't return copies. My initial filter cut down everything by about half! 
I need to look at this closer. However, preliminary results seem to be much improved. This is the clear next step for prompt engineering.

NOTE July 17th Update:
Dedupe complete at the bottom of this script.
"""

data_folder = "/home/msaad/workspace/honors-thesis/data-collection/data/"
responses_dict = pickle.load(open(data_folder + "scraper_output.p", "rb"))

# def get_text(item: tuple[str, requests.models.Response]) -> list:
#     """
#     Parses HTML for 'sentences', as described above.
#     """
#     key, response = item

#     # Your BeautifulSoup object
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Create an empty list to hold the sentences
#     long_texts = []

#     # Define custom punctuation
#     custom_punctuation = ',./?!:;$#&+*()"'

#     # Loop through all the strings in the BeautifulSoup object
#     for string in soup.stripped_strings:
#         # Consider a long text as a text having 15 or more words
#         if len(string.split()) >= 10: # NOTE July 17. Next run edit - 10 
#             # Check if the string contains only alphanumeric, whitespace and custom punctuation characters
#             if all(c.isalnum() or c.isspace() or c in custom_punctuation for c in string):
#                 # Check if the string contains more than 3 whitespaces in a row
#                 if not re.search(' {3,}', string):
#                     long_texts.append(string)

#     # NEW: Added so semantic search queries return bigger, more in context results.
#     # "chunk" is a string of all the long sentences (as described above) separated by a space.
#     chunk = " ".join(long_texts)
    
#     # OLD: return long_texts
#     return key, chunk

def get_text(item: tuple[str, requests.models.Response]) -> list:
    """
    Parse HTML using trafilatura. This is much simpler than the above method, with similar results.

    trafilatura is a very robust package for reading, and cleaning htmls.
    """
    key, response = item
    html = response.text

    cleaned_html = trafilatura.extract(
        filecontent = html, 
        include_tables=False
    )

    return key, cleaned_html

def clean_dict(responses_dict: dict[str, requests.models.Response]) -> dict[str, requests.models.Response]:
    """
    Indepth filtering of webpages/URLs. This is essential for semantic search especially since it will be prone to returning bad data if that's what you feed it in.
    """

    # Filter out to only successful requests (status code == 200)
    res_dict = {}
    for url, response in responses_dict.items():
        if response.status_code == 200:
            res_dict[url] = response

    urls = res_dict.keys()

    ## Many of the URLs have .html on he end of it. This causes some duplication where there is a url with a .html and another without.
    # This step filters them out, and standardizes it. If both exist, it'll remove the .html version, if only the .html version exists, it'll strip it and add it to the url list
    html_urls = set()
    non_html_urls = set()

    for url in urls:
        if url.endswith('.html'):
            stripped_url = url[:-5]  # strip .html
            html_urls.add(stripped_url)
        else:
            non_html_urls.add(url)

    urls = set(non_html_urls | (html_urls - non_html_urls))

    # Standardize the _ and - to -. Many copies of URLs with differences.
    urls = {url.replace('_', '-') for url in urls}

    # Filter length of URL?
    # urls = {url for url in urls if url.count('/') < 6}

    # Get rid of "live". Just wayyyyy too many of them, mostly professor websites.
    bad_list = ['brockport.edu/live/', 'brockport.edu/life/', 'archive', 'transfer-credit', 'research-foundation']
    urls = {url for url in urls if all(word not in url for word in bad_list)}


    # Important decision!! Going to limit URL length for all non-admissions/faid webpages. Just need to filter this down more... A lot more... 
    good_list = ['brockport.edu/admissions/', 'brockport.edu/academics/advisement/handbook']
    new_url = set()
    for url in urls:
        if any(word in url for word in good_list):
            new_url.add(url)
        elif url.count('/') < 5:
            new_url.add(url)
    urls = new_url

    # Make a dictionary of the URLs we've filtered it down to
    return_dict = {}
    for url in urls:
        # Not great way, but a way to do this. I removed all the .html ones, but still need to access their contents in dictionary.
        # I'll probably change this to make it cleaner in the future.
        if url in responses_dict.keys():
            return_dict[url] = responses_dict[url]
        else:
            # remember back when I changed all _ to -? Yeah... Need to wrap in try catch and see how this goes

            # NOTE This is kindof like a practice question. Figured chatgpt could do it, and sure enough, it could. Here is the prompt to understand what's going on:
            # ------
            # if i have a python string "a-b-c-d-e-f", write me a loop that will loop over ever permutation of _ being replaced for -. 
            # For example, "a-b-c_d_e-f", "a-b_c-d_e_f" will all be tested. I want to print every single permutation. Use code interpreter to verify your answer
            # ------
            # I just applied its answer here

            n = url.count('-')  # Number of "-" in the string

            # Loop over all binary numbers with 'n' bits
            for i in range(2**n):
                # Convert the number to binary and pad it with zeros on the left to get 'n' bits
                binary = format(i, f'0{n}b')
                
                # Replace "-" with "_" in the string according to the binary number
                new_url = url
                for j in range(n):
                    if binary[j] == '1':
                        new_url = new_url.replace('-', '_', 1)
                    else:
                        new_url = new_url.replace('-', '-', 1)
                        
                try:
                    return_dict[url] = responses_dict[url+'.html']
                except:
                    continue

    return return_dict


# Apply function to responses_dict
responses_dict = clean_dict(responses_dict)

# NOTE: Might want to consider saving off this dictionary to use in other parts of this project...


print("Begin fetching all the sentences...")
# Maps get_text over the dictionary to generate a list of key/chunk pairs.
all_sentences = list(filter(lambda x: x[1] != '', map(get_text, tqdm(responses_dict.items()))))
print(f"Saving off {len(all_sentences)} sentences.")

# Save off to a csv file
csv_name = "clean_chunks_from_html.csv"
with open(data_folder + csv_name, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['url', 'data'])
    for key, sentence in all_sentences:
        writer.writerow([key, sentence])


# # Now to split up the file into chunks for langchain
# NOTE July 20th update. Can be done with CSV. This is a waste unfortunately

# chunks_csv = pd.read_csv(data_folder + csv_name, names=['source', 'data'])

# # Mid July discovery! I have quite a bit of duplicate data, so this clears it out.
# chunks_csv = chunks_csv[~chunks_csv['data'].duplicated(keep='first')]
# chunks_store_path = f"{data_folder}vectordb_filestore/"

# # Check if path exists, if not, make it?
# if not os.path.exists(chunks_store_path):
#     response = input(f"\n{chunks_store_path} does not exist.\nWould you like to create it (y/n)? ")
#     if response in 'yY':
#         os.mkdir(chunks_store_path)
#     else:
#         print("Path not available. Cannot save off chunks.")
#         exit(1)

# for row_index in range(len(chunks_csv)):
#     # open the file with write mode
#     path = chunks_csv.iloc[row_index, 0].replace('https://www2.', '').replace('/', '_')
#     with open(f"{chunks_store_path}chunk_{path}.txt", 'w') as file:
#         # write a row of the csv to the file
#         file.write(chunks_csv.iloc[row_index, 1])

# print("\nWarning: chunks_from_html.csv still contains deplicated data. Only explicitly split chunks are deduped!")