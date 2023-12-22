"""
Script for cleaning scraped webpage data retreived in '1_scraping.py'

The filters we have applied throughout are static and highly specific to SUNY Brockport.
"""

from tqdm import tqdm
import pandas as pd
import trafilatura
import json

# Load in the raw data dictionary file. Recall that keys are URLs and values are HTML strings 
with open("data/raw_scraper_output.json", "r") as f:
    data = json.load(f)


def get_text(item: tuple[str, str]) -> list:
    """
    Parses and cleans raw HTML files using trafilatura. 

    Trafilatura is a very robust package for reading, and cleaning htmls. Using it 
    allows for a considerable simplification of the cleaning process. For more information
    refer to their documentation: https://trafilatura.readthedocs.io/en/latest/

    Args:
        item: A key-value pair from responses_dict stored in a tuple
    """
    key, html = item

    cleaned_html = trafilatura.extract(
        filecontent = html, 
        include_tables=False,
        deduplicate=True
    )

    return key, cleaned_html


print("\nBegin fetching full cleaned data... This will take a little time.")

# Get cleaned version of html for all data. Wrap in tqdm for progress bar.
df = pd.DataFrame(filter(lambda x: x[1] != '', map(get_text, tqdm(data.items()))), columns=['url', 'data'])

# Drop duplicates after cleaning HTMLs for any aliased URLs that may've gotten through.
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

    Why? /live/ is dominantly professor webpages that I don't want cluttering this dataset. 
    There are over 1000 of them, and the grand majority are just names and pictures with no 
    additional information. 
    """
    if url.startswith("https://www2.brockport.edu/live/"):
        return data.startswith("Policy Statement\n")
    else:
        return True

# Apply the function above to the dataframe
df = df[df.apply(lambda row: remove_excess_profile_stuff(row['url'], row['data']), axis=1)]

# Filter out webpages that only have a little bit of data (< 275 characters)
df = df[df['data'].str.len() > 275]








print("LENGTH BEFORE: ", len(df))

# Sort by data then standardize and drop duplicate URLs
# (This makes it so we prefer more data when deduping URLs)
df.index = df['data'].str.len()
df = df.sort_index(ascending=False).reset_index(drop=True)

# Standardization. Some URLs have multiple versions.
# Dropping "data" duplicates gets rid of 95% cases,
# but for categorizing there are some abnormalities with
# non-standardized URLs. Hence this standardization. 
df['url'] = df['url'].str.replace("_", "-")
df['url'] = df['url'].str.removesuffix(".html")
df = df.drop_duplicates(subset=['url'])

print("LENGTH AFTER: ", len(df))
print("NOTE TO SELF: IF LENGTHS ARE EQUAL THEN YOU CAN REMOVE THIS URL STUFF. I suspect it won't be needed anymore which would be great!")











# Resort by URL
df = df.sort_values(by=['url'])

# Drop rows where the URL contains any of the strings in the list
strings_to_remove = [
    '/transfer.credit/planning.guide', 
    '/archive/', 
    '/archives/', 
    'edu/go', 
    'edu/info', 
    'edu/quick.links', 
    '/advancement.communications'
]
df = df[~df['url'].str.contains('|'.join(strings_to_remove), regex=True)]


print(f"Saving off full, cleaned dataset of {len(df)} webpages.")

df.to_csv("data/website_data.csv", index=False)

print("\nDone!")