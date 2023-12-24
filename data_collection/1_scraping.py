"""
This script handles all the raw data collection from the SUNY Brockport website.

Simply put, we start at a webpage (brockport.edu for this project) and search for
all the links on that page, and then visit those. We do this recursively where each
of these cycles (going back and starting again) is considered 1 depth. Higher depth
will visit more webpages, and an arbitrarily high depth should (in theory) visit
all webpages.

This is written to avoid scraping the same page multiple times, and to obey the
websites robots.txt file. 
"""

import requests
import time
import json
import os
import random
import warnings
from tqdm import tqdm
from bs4 import BeautifulSoup

# ARGUMENTS
DEPTH = 10   # Search further for links. Higher = more data, but takes more time. See README.md for more info
WEBPAGE = "https://www2.brockport.edu" # Base webpage to search from, changing this might break disallow section
SAVE_PATH = "data/raw_scraper_output.json" # Save location

if not os.path.isdir(os.path.dirname(SAVE_PATH)):
    print("Invalid save path. Please provide a valid path.\n")
    print("I suggest you make sure that you are currently inside the data_collection directory.")
    exit()

# ANSI escape codes for colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
ENDC = '\033[0m'

# ----------------------------------------------------------------
def create_disallow_list() -> list:
    """
    This function fetches and processes the robots.txt file of the SUNY Brockport website. It builds a dictionary
    containing lists of 'Allowed' and 'Disallowed' URLs as per the directives in the robots.txt file. In this case,
    we are only concerned with the 'Disallowed' URLs since the SUNY Brockport website does not specify an allow list.

    After parsing the 'Disallow' directives from the robots.txt file, additional URLs that tend to cause issues are 
    manually appended to the disallowed list. These include links to certain files, search pages, and various social media 
    platforms.

    The function uses the curl command to fetch the robots.txt file and regular expressions to parse the contents.

    Returns:
    list: A list of disallowed URLs, or portions of URLs, including manually appended URLs.
    """

    # Fetch the robots.txt file content
    result = os.popen("curl https://www2.brockport.edu/robots.txt").read()
    result_data_set = {"Disallowed":[], "Allowed":[]}

    for line in result.split("\n"):
        if line.startswith('Allow'):    # this is for allowed url
            result_data_set["Allowed"].append(line.split(': ')[1].split(' ')[0])    # to neglect the comments or other junk info
        elif line.startswith('Disallow'):    # this is for disallowed url
            result_data_set["Disallowed"].append(line.split(': ')[1].split(' ')[0])    # to neglect the comments or other junk info

    disallow_list = result_data_set['Disallowed']

    # Adding my own filters. These files tend to not work throughout.
    disallow_list.extend(['/live/files/', '.html/', 'livewhale', '/bounce/', 'instagram.com', 'twitter.com', 'facebook.com', 'youtube.com', '/search/'])

    return disallow_list

# Create the disallow list
if WEBPAGE == "https://www2.brockport.edu":
    disallow_list = create_disallow_list()
else:
    warnings.warn("Robots.txt filtering is NOT supported for non 'https://www2.brockport.edu' webpages! Consider updating this section...")
    disallow_list = []

# Scraping Functions---------------------------------------------------
# To help simplify the process of scraping the website, these functions 
# are defined to not overlap different stages of the scraping process

def get_webpages(urls_to_get: list, urls_tried_already: list) -> tuple[dict, list]:
    """
    This function scrapes the content from a list of webpage links. Each page content is stored in a dictionary
    with the link is the key. If there is an issue while processing a link, it catches the exception, displays 
    an error message, and continues with the next link. To prevent overwhelming the server, a random sleep
    between 1 to 3 seconds is implemented between each request.

    Parameters:
    urls_to_get (list): A list of webpage URLs to scrape.
    urls_tried_already (list): A list of URLs that have already been scraped to prevent duplicate requests.

    Returns:
    dict: A dictionary with URLs as keys and the corresponding webpage content as values.
    """
    assert(isinstance(urls_to_get, list))
    assert(isinstance(urls_tried_already, list))

    data = {}
    for url in tqdm(urls_to_get):
        if url in urls_tried_already:
            continue
        urls_tried_already.append(url)
        time.sleep(random.uniform(1, 3))
        try:
            with requests.get(url) as response:
                if response.status_code == 200:
                    data[response.url] = response.text
                    tqdm.write(f"{GREEN}Success! URL: {url}{ENDC}")
                else:
                    tqdm.write(f"{YELLOW}Bad status code: {response.status_code} for {url}{ENDC}")
        except Exception as e:
            tqdm.write(f"{RED}An error occurred when processing {url}{ENDC}")
            time.sleep(2)
            pass

    return data, urls_tried_already

def get_links_from_webpage(response) -> list:
    """
    This function processes a response object from a webpage, typically obtained via a requests.get call. It uses
    BeautifulSoup to parse the webpage HTML, extracts all links on the page and filters these to include only those
    pointing to other pages within the same website. It also excludes links leading to the website's robots.txt file
    and certain other disallowed files. It removes duplicate links, and returns a list of unique, filtered URLs.
    Finally, "https://www2.brockport.edu" is appended to the beginning of each link to generate the full URL.

    Parameters:
    responseObj: A requests.Response object containing the webpage's content.

    Returns:
    list: A list of unique and filtered URLs from the webpage.
    """
    links = []
    soup = BeautifulSoup(response, 'html.parser')

    # Filter the links to get only those that point to other pages realtive to its position.
    # This indirectly ensures that all links are on the main brockport.edu website.
    links = soup.find_all('a')
    links = [link for link in links if link.get('href') is not None]
    links = [link.get('href') for link in links if link.get('href').startswith('/') and '#' not in link.get('href')]

    # Filter out any disallowed links
    links = [link for link in links if not any(bad_link in link for bad_link in disallow_list)]
    links = [link.rstrip('/') for link in links]

    # Remove duplicates
    links = list(set(links))

    # Create full URL
    links = ["https://www2.brockport.edu" + link for link in links]

    return links

def recursive_scrape(webpage: str, depth: int, start: dict = {}) -> dict:
    """
    This function conducts a recursive scrape of a given webpage up to a specified depth. It begins by scraping the
    base webpage and collecting all unique links from it. Then, for each depth level, it visits the previously
    collected links that have not been visited yet, scrapes their content and collects new links from these pages.
    The process is repeated until the specified depth is reached. The function maintains a dictionary of visited
    links (with the link as the key and the page content as the value), which is returned as the output.

    Parameters:
    webpage (str): The base URL from where the scrape begins.
    depth (int): The maximum depth of the recursion, i.e., how many levels deep the function will scrape from the base page.
    start (dict): A dictionary of previously visited links and their corresponding page content. This is used to continue. Note about starting: Since we only store the "real URL", aliased URLs we were keeping track of are lost. This means that for every alias we find on the stored webpages, they'll be scraped again. This leads to a  *significant* amount of time is wasted, but is still much faster than scraping everything again. Could be fixed if this funtionality was rewritten.

    Returns:
    dict: A dictionary with URLs as keys and the corresponding webpage content as values.
    """
    links_to_visit = [webpage]
    links_visited = set()
    data = start

    for level in range(1, depth+1):
        print('\n\nDepth level', level, "start...")
        
        # visit only links we haven't searched before
        links_to_visit = [link for link in links_to_visit if link not in links_visited]

        # add all the links_to_visit to the data
        webpages_dict, visited = get_webpages(links_to_visit, list(links_visited))

        data.update(webpages_dict)
        links_visited.update(visited)

        # update the links to visit
        links_to_visit = []
        for response in data.values():
            links_to_visit.extend(get_links_from_webpage(response))

        # save intermediate results
        with open(f"data/scraper_intermediate/depth_{level}.json", 'w') as f:
            f.write(json.dumps(data, indent=4))

    return data

# --------------------------------------------------
# Start running this thing and save it off as a JSON

data = recursive_scrape(webpage=WEBPAGE, depth=DEPTH)
# data = {url: res for url, res in data.items() if res.status_code == 200}

with open(SAVE_PATH, 'w') as f:
    f.write(json.dumps(data, indent=4))