"""
Creating the question/answer dataset from the cleaned data

We are using GPT-4 in this implementation to create questions for our dataset.
This script parallizes 64 requests using python's asyncronous function support at
a time to do this. We find that a rate limit of 64 requests at a time is around 
our allowed limit, but this can change depending on prompt used, etc.

Async requests with asyncio

Inspiration: https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio
"""

from bs4 import BeautifulSoup
import re
import os
import openai
import pickle
import time
import json
import random
from itertools import chain
import concurrent.futures
import requests
import pandas as pd
import asyncio
import httpx
import time

data = pd.read_csv("data/website_data.csv")
openai.api_key = os.getenv("OPENAI_API_KEY")

system = """You are a helpful question maker for SUNY Brockport. Given some information about the school, your job is to
parse that information and generate realistic questions a prospective student or faculty member might ask. For instance,
"How can I apply to financial aid?" is a question a student could reasonably ask. Generate up to five of such questions and
respond in JSON format, where "question" is one field, and "answer" is another."""

prompt = lambda content: f"""Based on the content given generate questions. Think about your answer carefully before
resonding, and be sure to answer in JSON format."""

MODEL = "gpt-4-1106-preview"
N_CONCURRENT = 64
URL_MESSAGES = [(url, [
    {"role": "system", "content": system},
    {"role": "user", "content": prompt(content)}
]) for url, content in zip(data["url"], data["data"])]

start_time = time.time()

import asyncio
import random

MAX_RETRIES = 3  # Maximum number of retries
BASE_WAIT_TIME = 2  # Base wait time in seconds

async def async_openai_chat_call(client, messages, url, model, retries=0):
    async_start_time = time.time()
    try:
        response = await client.post(
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                     'Content-Type': 'application/json'},
            json={'model': model, 'messages': messages}
        )
        response.raise_for_status()  # Raise an exception for HTTP error codes
        response = response.json()
        duration = time.time() - async_start_time
        print(f"Success! Complete in {duration:.2f}s with {response['usage']['total_tokens']} tokens for {url}")
        return response
    except Exception as e:
        print(e)
        if retries < MAX_RETRIES:
            wait_time = BASE_WAIT_TIME * (2 ** retries)  # Exponential backoff
            print(f"Error for {url}: {e}. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            return await async_openai_chat_call(client, messages, url, model, retries + 1)
        else:
            print(f"Failed after {MAX_RETRIES} retries for {url}: {e}")
            return None  # Or handle the failure in some other way

# Rest of your code...


# async def async_openai_chat_call(client, messages, url, model):
#     async_start_time = time.time()
#     response = await client.post(
#         'https://api.openai.com/v1/chat/completions',
#         headers={
#             'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
#             'Content-Type': 'application/json'
#         },
#         json={
#             'model': model,
#             'messages': messages
#         }
#     )
#     response = response.json()
#     duration = time.time() - async_start_time

#     print(f"Success! Complete in {duration:.2f}s with {response['usage']['total_tokens']} tokens for {url}")

#     return response

async def gather_with_limit(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros))

async def make_concurrent_chat_calls(messages_list, n_concurrent, model):
    async with httpx.AsyncClient() as client:
        tasks = [async_openai_chat_call(client, messages, url, model) for url, messages in messages_list]
        responses = await gather_with_limit(n_concurrent, *tasks)
        return responses
    
async def main():
    responses = await make_concurrent_chat_calls(
        messages_list=URL_MESSAGES, 
        n_concurrent=N_CONCURRENT, 
        model=MODEL
    )
    # Process responses...
    print(responses)

asyncio.run(main())

print("--- %s seconds ---" % (time.time() - start_time))










exit()

from bs4 import BeautifulSoup
import re
import os
import openai
import pickle
import time
import json
import random
from itertools import chain
import concurrent.futures
import requests

"""Load in data"""

test = True # sample or run on full dataset?
sample_size = 4 # if test=True

scape_output = pickle.load(open('data/scraper_output.p', 'rb'))
openai.api_key = os.getenv("OPENAI_API_KEY")

if test: 
    keys = random.sample(list(scape_output.keys()), sample_size)
    scape_output = {key: scape_output[key] for key in keys}

# # Things to note.
# 
# This spot specifically has room for improvement. It might be worthwhile to implement changes to provide improvements to the dataset generation. More recently, the paper [Textbooks Are All You Need](https://arxiv.org/pdf/2306.11644.pdf) suggests that data quality may infact be the key to success in projects like this. Knowing that, here are some ideas to improve the data generation process in this project.
# 
# 1. Clean up context provided to GPT more. Not much effort was put in to this step under the presupposition that GPT would be able to interpret it regardless. In practice, that did turn out to be true. However, it is difficult to guage how much of an effect, if any, the excess text from buttons and such made on the quality of responses. It is possible by removing it, and instead feeding in something similar to the chunks generated for the vector database (see [relevant code](./wrangling%20for%20vectordb.py)) we could get better quality responses. This could also create more diverse questions, or at least less similar ones. I hypothesize that since at the bottom of each page there is contact info, if GPT struggles to reach its target question count on short webpages, it could easily be writing questions regarding contact info many times.
# 2. Use GPT4 for higher quality questions.
# 3. Use more targetted webpages. I.e. avoid professor webpages, and try to keep it as general as possible.
# 

def process_item(key: str, value: requests.models.Response, question_count: int) -> tuple[str, str]:
    """
    Generates a set of questions based on the HTML provided. Not much effort was put into cleaning
    the HTMLs before passing them in to gpt3.5 (theres a lot of excess words from buttons and such);
    the idea is that gpt3.5 will handle the excess and still return a good result. In practice, this
    has seemed to work, but there could be room for improvement in question generation by improving
    this part specifically.

    Args:
        key: URL of the response webpage
        value: Response object from scraping
        question_count: How many questions should gpt3.5 strive to reach
    """
    start_time = time.time()

    # Clean up context to give gpt
    soup = BeautifulSoup(value.content, "html.parser")
    context = re.sub('[\n]+', '\n', soup.text.strip())

    # Main portion of the prompt
    prompt = f"""Based on the cleaned HTML given below, generate as many questions possible with their answers.
    Try to make the questions relevant from the perspective of a prospective or current student, as well as faculty and staff.
    Format your responce in JSON, with the "instruction" field containing the question, an empty "input" field, and the answer in the "output" field.
    Include up to {question_count} questions, each being a sentence or two long in length. Do not include question number.
    Keep answers somewhat brief, but be enthusiastic in your response!\n\n"""
    
    # API call
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", # Try GPT-4?
        messages=[
            {"role": "system", "content": "You are a helpful question answer generator."},
            {"role": "user", "content": f"{prompt}\n This is the cleaned HTML: \n{context}\n Start:."},
        ]
    )

    qa = completion.choices[0].message.content
    tokens = completion.usage["total_tokens"]

    print(f"Success! Complete in {time.time() - start_time:.2f}s with {tokens} tokens for {key}")

    return key, qa

# # Parallelize Requests
# 
# Since he ChatCompletion API does not yet have support for batching, sending in many requests is difficult. To overcome this, the step below parallelizes the request. Basically, up to 64 requests will go through concurrently, and print out as they are completed. The number 64 is chosen in this case since it seems to utilize the majority of my request limit, but does not go over. If this is ever used for other projects, you may consider adjusting this number given the size of the prompt relative to the one used here.


gpt_output = {}
# How many questions should GPT aim to make per webpage. Default: 25
question_count = 25

with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
    future_to_item = {executor.submit(process_item, key, value, question_count): key for key, value in scape_output.items()}
    for future in concurrent.futures.as_completed(future_to_item):
        key = future_to_item[future]
        try:
            key, qa = future.result()
            gpt_output[key] = qa
        except Exception as exc:
            print('%r generated an exception: %s' % (key, exc))


pickle.dump(gpt_output, open('data/gpt_output.p', 'wb'))


# # Parse GPT output to JSON
# 
# While the prompt above to translate clean HTML to question/answer format does specify to do it in JSON format, GPT3.5 does not always do it perfectly. However, it always get close. Instead of trying to fix the JSON output from GPT, In this step I'm using regex to parse for all the instructions (questions), and outputs (answers). This is returned into a python list of dictionaries, which is appended for each webpage. Eventually I shuffle this so the questions are all mixed up instead of grouped by webpage, and dump it to a json file.
# 
# For any questions which seem off, investigate the original webpage. I've left gpt_output as a python dictionary specifically for this reason, so that we can always refer back to the data and know exactly where it came from.


def generate_json(gpt_output: dict[str, str], filename: str) -> None:
    """
    Parses GPT output into a JSON file. This function uses regex to parse for all the instructions (questions), and 
    outputs (answers) from the GPT output which is a dictionary. The parsed data is returned into a Python list of 
    dictionaries, which is appended for each webpage. This list is shuffled to mix up the questions and then dumped 
    to a JSON file.

    Args:
        gpt_output (dict): The dictionary containing the GPT output.
        filename (str): The JSON filename to write the parsed and shuffled list of dictionaries to.
    """

    # The regular expression pattern for a JSON object with "instruction" and "output"
    pattern = r'"instruction":\s*"(.*?)",.*?"output":\s*"(.*?)"'

    def extract_data(s):
        matches = re.findall(pattern, s, flags=re.DOTALL)
        # Add a conditional filter in the list comprehension
        data = [{"instruction": m[0], "output": m[1]} for m in matches if m[0] and m[1] and '"' not in m[0] and '"' not in m[1]]
        return data

    jsonqa = []

    for value in gpt_output.values():
        clean_value = extract_data(value)
        jsonqa.append(clean_value)

    jsonqa = list(chain(*jsonqa))

    random.shuffle(jsonqa)

    # Write to a JSON file
    with open('data/' + filename + '.json', 'w') as f:
        json.dump(jsonqa, f, indent=4)  # Dump the entire list at once


gpt_output = pickle.load(open('data/gpt_output.p', 'rb'))

generate_json(gpt_output, "full_dataset_v4")
