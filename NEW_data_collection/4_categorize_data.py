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
This script is responsible for overseeing that process by creating the groups.

The output of this script is a csv file that has the following columns:

1. category: The category of the data
2. subcategory: The subcategory of the data
3. url: The url of the data
4. data: The data itself

The subcategory is only assigned if there are more than 5 unique URLs in the subcategory.

!! This script is dependent on the cleaned data, which is the output of 2_cleaning.py

------------------------------------------------------------------------------------
# The How.

Most of this will be done based off of the URL. I've done a lot of exploratory analysis
on the data, and the URL is the best way to categorize the data -- they contain important info!
"""

import pandas as pd

# Cleaned data from 2_cleaning.py
df = pd.read_csv("data/website_data.csv")

SAVE_NAME = "categorized_data.csv"  # Name of the file to save the data to
SAVE_LOCATION = "data/"             # Location to save the data

# ------------------ START OF SCRIPT -------------------

# List to hold the categorized data
categorized_data = []

# Loop through each row in the DataFrame
for _, row in df.iterrows():
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
            categorized_data.append({
                "category": category,
                "subcategory": subcategory,
                "url": url,
                "data": data
            })
        else:
            categorized_data.append({
                "category": category,
                "subcategory": None,
                "url": url,
                "data": data
            })
    else:
        categorized_data.append({
            "category": category,
            "subcategory": None,
            "url": url,
            "data": data
        })

# Convert the list of dictionaries to a DataFrame
categorized_df = pd.DataFrame(categorized_data)

# Save the DataFrame to a CSV file
categorized_df.to_csv(SAVE_LOCATION + SAVE_NAME, index=False)

print("Complete!")