"""
This script handles all the raw data collection from the SUNY Brockport website.

It also includes support for honoring the websites disallow list written in its robots.txt
file. The output from this script is a dictionary where the value is the response object
from any given webpage, and the key is the URL of that webpage. This is done to allow us
to easily be able to go back in time and figure out where information is coming from,
and also allows us to put advanced filter layers on based on the information in the URL
"""

import requests
import time
import pickle
import os
import random
from bs4 import BeautifulSoup

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
disallow_list = create_disallow_list()

# Scraping Functions---------------------------------------------------
# To help simplify the process of scraping the website, these functions 
# are defined to not overlap different stages of the scraping process

def search_page(links: list) -> dict:
    """
    This function scrapes the content from a list of webpage links. Each page content is stored in a dictionary
    with the link itself serving as the key. If there is an issue while processing a link, it catches the exception,
    displays an error message, and continues with the next link. To prevent overwhelming the server, a random sleep
    between 1 to 3 seconds is implemented between each request.

    Parameters:
    links (list): A list of webpage URLs to scrape.

    Returns:
    dict: A dictionary with URLs as keys and the corresponding webpage content as values.
    """
    data = {}

    for link in links:
        try:
            data[link] = requests.get(link)
            time.sleep(random.uniform(1, 3))
        except Exception as e:
            print(f"An error occurred when processing {link}")
            continue

    return data

def get_links_from_webpage(responseObj) -> list:
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

    soup = BeautifulSoup(responseObj.text, 'html.parser')

    # Find all the links on the page
    links = soup.find_all('a')

    # Filter the links to get only those that point to other pages on the same site
    # This is done by checking if the link starts with '/' (indicating it's a relative link)
    # and doesn't contain '#' (which would indicate it's a link to a specific part of the same page)
    links = [link for link in links if link.get('href') is not None]
    links = [link.get('href') for link in links if link.get('href').startswith('/') and '#' not in link.get('href')]

    # APPLY FILTER... Gets rid of robots.txt files, and some others.
    links = [link for link in links if not any(bad_link in link for bad_link in disallow_list)]

    links = [link.rstrip('/') for link in links]

    # Remove duplicates
    links = list(set(links))

    links = ["https://www2.brockport.edu" + link for link in links]

    return links

def recursive_scrape(webpage: str, depth: int) -> dict:
    """
    This function conducts a recursive scrape of a given webpage up to a specified depth. It begins by scraping the
    base webpage and collecting all unique links from it. Then, for each depth level, it visits the previously
    collected links that have not been visited yet, scrapes their content and collects new links from these pages.
    The process is repeated until the specified depth is reached. The function maintains a dictionary of visited
    links (with the link as the key and the page content as the value), which is returned as the output.

    Parameters:
    webpage (str): The base URL from where the scrape begins.
    depth (int): The maximum depth of the recursion, i.e., how many levels deep the function will scrape from the base page.

    Returns:
    dict: A dictionary with URLs as keys and the corresponding webpage content as values.
    """
    links_to_visit = [webpage]
    data = {}

    for interation in range(depth):
        print('Pass', interation)
        
        # visit only links we haven't searched before
        links_to_visit = [link for link in links_to_visit if link not in list(data.keys())]
        links_to_visit = list(set(links_to_visit))

        # add all the links_to_visit to the data
        data.update(search_page(links_to_visit))

        # reset links_to_visit
        links_to_visit = []

        # update the links to visit
        for response in data.values():
            links_to_visit.extend(get_links_from_webpage(response))

    return data

# Using the Scraper ------------------------------------------------------
# Refer to README to get expectations for time spent scraping the website.

# I've chosen a depth of 5, which should cover the majority of webpages on the website. 
# For future runs, I may up that number. A depth of 6 should add roughly an additional 1000 webpages.

data = recursive_scrape("https://www2.brockport.edu", 5)

pickle.dump(data, open('data/scraper_output.p', 'wb'))