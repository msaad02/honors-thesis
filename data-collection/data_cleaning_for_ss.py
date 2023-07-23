"""
Script for cleaning and filtering scraped webpage data. This is used for creating data
for the semantic search engine to use.

This script performs two main tasks: cleaning raw HTML files and applying 
in-depth filtering on a dictionary of webpage data. The raw HTML files are 
parsed and cleaned using the trafilatura library, a robust tool for handling HTML data. 

For the dictionary of webpage data (where the key is a URL and the value is the response 
from scraping the URL), the script applies multiple filtering steps. These steps include 
removing unsuccessful requests, standardizing URL formats, and removing unwanted websites. 
Special conditions are also applied for specific URL patterns to provide more relevant results. 

The script operates on a pickled dictionary (named "scraper_output.p") loaded from the 
"scraper_output.p" file in a predefined data folder. The cleaned and filtered data is then 
stored as a CSV file in the same data folder.

This script is specifically tailored for a particular set of web content, and modifications may 
be necessary for use with other data.
"""

import pickle
import requests
from tqdm import tqdm
import csv
import trafilatura

data_folder = "/home/msaad/workspace/honors-thesis/data-collection/data/"
responses_dict = pickle.load(open(data_folder + "scraper_output.p", "rb"))


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

def clean_dict(
    responses_dict: dict[str, requests.models.Response]
) -> dict[str, requests.models.Response]:
    """
    Indepth filtering of webpages/URLs.
    
    This is essential for semantic search especially since it will be 
    prone to returning bad data if that's what you feed it in.

    Args:
        Dictionary where key is url, and value is response from scraping url

    Return:
        Cleaned dictionary, in the same form of input dictionary.
    """

    # Filter out to only successful requests (status code == 200)
    res_dict = {}
    for url, response in responses_dict.items():
        if response.status_code == 200:
            res_dict[url] = response

    urls = res_dict.keys()

    # Many of the URLs have .html on he end of it. This causes some duplication 
    # where there is a url with a .html and another without. This step filters 
    # them out, and standardizes it. If both exist, it'll remove the .html version, 
    # if only the .html version exists, it'll strip it and add it to the url list
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
    # This likely should be done by editting the dictionary directly,
    # but I got a workaround for calling the old urls later in this function.
    urls = {url.replace('_', '-') for url in urls}

    # Getting rid of 'bad' websites, these account for a tremendous portion
    # of the data, and offer little to no information a user might need # 'transfer-credit'
    bad_list = ['brockport.edu/live/', 'brockport.edu/life/', 'archive', 'research-foundation'] 
    urls = {url for url in urls if all(word not in url for word in bad_list)}

    # --------------------------------------------------------------------------------
    # NOTE: By this point, the dataset is still wayy too big. Primarily these step from each
    # departments information. For example, computer science has like 20-50 webpages.
    # This trend follows for other majors. So in this next step I am limiting the URL
    # length, as denoted by the number of /'s in the URL. 5 is selected since it allows
    # you to reach the base webpage for most if not all departments, but does not allow
    # any further. I don't want this functionality for all webpages though. Admissions for
    # instance has many useful webpages (and not an absurd amount) exceeding this limit, so
    # I explicitly 'green-lit' it pass this filter. I've also done so with advisement/handbook,
    # Since these crucical webpages provide important high level overviews of every major.
    # With these changes, a typical search result about a specific major will return it's
    # handbook page, and it's main webpage (for example, https://www2.brockport.edu/academics/computing-sciences/)
    # So far this has worked with good results. However, it is very subject to change.
    # --------------------------------------------------------------------------------

    # Important decision!! 
    good_list = ['brockport.edu/admissions/', 'brockport.edu/academics/advisement/handbook']
    new_url = set()
    for url in urls:
        if any(word in url for word in good_list):
            new_url.add(url)
        elif url.count('/') < 5:
            new_url.add(url)
    urls = new_url

    # Makes a final dictionary with filters applied throughout
    # Aside from the first check for the status code, all the filters have applied
    # on the URL level only. This next (and final) step is to remake a new dictionary
    # with the URL and status code filters applied. 

    # This got somewhat complex, and is squarly in "it works" category. Ain't broke- not fixing it.
    return_dict = {}
    for url in urls:
        # Not great way, but a way to do this. I removed the .html from many urls, 
        # but still need to access their contents in the original dictionary (with .html)

        # Check if url in responses_dict. (Most go in here)
        if url in responses_dict.keys():
            return_dict[url] = responses_dict[url]

        # Enters if I changed something throughout. (removing .html for instance, or changing _ to -)
        else:
            # remember back when I changed all _ to -? 
            # Yeah... Need to wrap in try catch and see how this goes

            # NOTE This is kindof like a leetcode question. 
            # I figured chatgpt could do it, and sure enough, it could. 
            # Here is the prompt to understand what's going on:
            # ------------------------------------------------------------
            # if i have a python string "a-b-c-d-e-f", write me a loop that 
            # will loop over ever permutation of _ being replaced for -. 
            # For example, "a-b-c_d_e-f", "a-b_c-d_e_f" will all be tested. 
            # I want to print every single permutation. Use code interpreter to verify your answer
            # ------------------------------------------------------------

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


# Apply cleaning to responses_dict
# NOTE: Might want to consider saving off this dictionary to use in other parts of this project...
responses_dict = clean_dict(responses_dict)

print("Begin fetching cleaned data...")
# Get cleaned version of html for all data. Wrap in tqdm for progress bar.
data = list(filter(lambda x: x[1] != '', map(get_text, tqdm(responses_dict.items()))))
print(f"Saving off {len(data)} cleaned data.")

# Save off to a csv file
csv_name = "cleaned_data.csv"
with open(data_folder + csv_name, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['url', 'data']) # Column names
    for key, sentence in data:
        writer.writerow([key, sentence])