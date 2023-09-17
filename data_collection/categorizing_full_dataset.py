"""
This script is part of a larger idea of creating a decision tree so at question time
we can  determine which dataset to use. For example, if the user asks a question about
financial aid, the decision tree will determine that we should use the dataset about
financial aid, or mapping admissions questions to admission specific datasets, etc.

The goal of this is to avoid semantic search returning unrelated results.
-------------------------------------------------------------------------

So, what specifically does this script do?

Here, we are categorizing the data into different datasets. Like I mentioned with the
financial aid example, we want to be able to map certain questions to certain datasets.
This script is responsible for overseeing that process by creating the datasets.

The vision for the output of this script is a json file that looks like this:

{
    admissions: {
        "url1": "data1",
        "url2": "data2",
        ...
    },
    financial_aid: {
        "url1": "data1",
        "url2": "data2",
        ...
    },
    academics: {
        "major1": {
            "url1": "data1",
            "url2": "data2",
            ...
        },
        "major2": {
            "url1": "data1",
            "url2": "data2",
            ...
        },...
    },...
}

If done correctly, we should be able to map any question to a specific dataset meaningfully,
and recursively (i.e. a question about a specific major will be firsted pipe through academics, etc.)

Importantly, this script is not responsible for cleaning the data. That is done in
full_dataset_cleaning.py. This script is only responsible for categorizing the data.

------------------------------------------------------------------------------------
# The How.

Most of this will be done based off of the URL. I've done a lot of exploratory analysis
on the data, and the URL is the best way to categorize the data -- they contain important info!
"""

""" ADDITIONALLY
NOTE: The idea is to break down the URLs and leverage their information to 
categorizethe data. Only categorize them in the way described by academics 
-> major1 or major2 way IF there are a lot of different categories inside 
of them, AND if each category contains more relevant inforamtion.

For instance, the major example came up because admissions/major1/... has a LOT 
of webpages. I don't want to mix those webpages up with admissions/major2. But 
if there aren't many webpages, say less than 5, then it's not worth doing.
"""


from collections import defaultdict
import pandas as pd
import json

# Load the data
df = pd.read_csv("/home/msaad/workspace/honors-thesis/data_collection/data/full_cleaned_data.csv")

# Initialize a nested dictionary to hold the categorized data
categorized_data = defaultdict(lambda: defaultdict(dict))

# Loop through each row in the DataFrame
for idx, row in df.iterrows():
    url = row['url']
    data = row['data']
    
    # Split the URL into segments
    segments = url.split("/")[3:]  # Skip the 'https://www2.brockport.edu'
    
    # Get the category and subcategory based on URL segments
    category = segments[0] if len(segments) > 0 else "other"
    subcategory = segments[1] if len(segments) > 1 else None
    
    # If subcategory exists and has sufficient unique URLs, categorize accordingly
    if subcategory:
        subcat_count = len(df[df['url'].str.contains(f"{category}/{subcategory}/")])
        if subcat_count > 5:
            categorized_data[category][subcategory][url] = data
        else:
            categorized_data[category][url] = data
    else:
        categorized_data[category][url] = data

# Convert to a regular dictionary and output to JSON
final_output = json.dumps(dict(categorized_data), indent=4)

# Save the JSON structure to a file
json_file_path = '/home/msaad/workspace/honors-thesis/data_collection/data/categorized_data.json'
with open(json_file_path, 'w') as f:
    f.write(final_output)

json_file_path
