"""
Script for cleaning scraped webpage data. This is used for creating data
for the semantic search engine to use.

** READ THIS **
------------------------------------------------------------------------------------
NOTE: You may be wondering what exactly the difference is between this script and
filtered_dataset_cleaning.py. The difference is that this script does not apply
any filtering to the data. It only cleans the raw HTML files. This is useful for
creating a large dataset for semantic search to use. The filtered_dataset_cleaning.py
script is used for creating a smaller dataset that has a more narrow scope, with
the goal of providing higher quality results over quantity.

This script is part of a larger idea of creating a decision tree so at question time
we can  determine which dataset to use. For example, if the user asks a question about
a specific major, the decision tree will determine that we should use the filtered
dataset about that major, etc. etc. To do this, we need access to as much data as
possible, hence the need for this script. At a high level, this script will do the
minimum amount of cleaning necessary to make the data usable for semantic search.
------------------------------------------------------------------------------------
"""

import pickle
import requests
from tqdm import tqdm
import trafilatura
import pandas as pd

data_folder = "/home/msaad/workspace/honors-thesis/data-collection/data/"

# NOTE: data_dictionary is a URL/response dictionary. The key is the URL and the value is the response from scraping the URL.
data_dictionary = pickle.load(open(data_folder + "scraper_output.p", "rb"))

# Filter out failed Request by keeping successful requests (Status code == 200)
data_dictionary = {url: response for url, response in data_dictionary.items() if response.status_code == 200}

def get_text(item: tuple[str, requests.models.Response]) -> list:
    """
    Parses and cleans raw HTML files using trafilatura. 

    Trafilatura is a very robust package for reading, and cleaning htmls. Using it 
    allows for a considerable simplification of the cleaning process. For more information
    refer to their documentation: https://trafilatura.readthedocs.io/en/latest/

    Args:
        item: A key-value pair from responses_dict stored in a tuple
    """
    key, response = item
    html = response.text

    cleaned_html = trafilatura.extract(
        filecontent = html, 
        include_tables=False,
        deduplicate=True
    )

    return key, cleaned_html


print("\nBegin fetching full cleaned data... This will take a little time.")

# Get cleaned version of html for all data. Wrap in tqdm for progress bar.
df = pd.DataFrame(filter(lambda x: x[1] != '', map(get_text, tqdm(data_dictionary.items()))), columns=['url', 'data'])

# Drop duplicates after cleaning HTMLs. This removes ~1.5k rows.
df = df.sort_values(by=['url'])
df = df.drop_duplicates(subset=['data'])


# Filtering out the /live/ section
# NOTE: /live/blurb/ might be worth keeping? It's a little different than the rest of /live/
def remove_excess_profile_stuff(url: str, data: str) -> bool:
    """
    There's a lot of bad stuff in /live/ urls. I went through it all, and really the
    only useful part of it is the policy information. This function returns TRUE if the url
    is either NOT a part of /live/profile, OR if it is a part of /live/profile, but contains
    "Policy Statement\n" in the data column. Otherwise, it returns False.
    """
    if url.startswith("https://www2.brockport.edu/live/"):
        return data.startswith("Policy Statement\n")
    else:
        return True

# Apply the function above to the dataframe
df = df[df.apply(lambda row: remove_excess_profile_stuff(row['url'], row['data']), axis=1)]


# Drop rows where the URL contains any of the strings in the list
strings_to_remove = ['/transfer-credit/planning-guide/', '/archive/', '/archives/']
df = df[~df['url'].str.contains('|'.join(strings_to_remove))]


print(f"Saving off full, cleaned dataset of {len(df)} webpages.")

df.to_csv(data_folder + "full_cleaned_data.csv", index=False)

print("\nDone!")