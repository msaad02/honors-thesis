"""
Creating the question/answer dataset from the cleaned data

We are using GPT-4 in this implementation to create questions for our dataset.
This script parallizes 64 requests using python's asyncronous function support at
a time to do this. We find that a rate limit of 64 requests at a time is around 
our allowed limit, but this can change depending on prompt used, rate limit, etc.
This could much more easily be done with a for loop, but this is significantly 
faster. Who likes to wait?
"""

import pandas as pd
import os
import time
import json
import asyncio
import httpx
import time

data = pd.read_csv("data/website_data.csv")

system = """You are a helpful question maker for SUNY Brockport. Given some information about the school, your job is to parse that information and generate realistic questions a prospective student or faculty member might ask. For instance, "How can I apply to financial aid?" is a question a student could reasonably ask. Generate up to five of such questions and respond in JSON format, where "question" is one field, and "answer" is another."""

# If changing prompt I highly recommend keeping the JSON specification the same. It makes parsing much easier, 
# ESPECIALLY for GPT-3.5. From my testing, GPT-4 usually outputs this format regardless.
prompt = lambda content: """Based on the content given generate questions. Think about your answer carefully before
responding, and be sure to answer in JSON format starting with \n```json\n[\n{\n    "question": "...",\n"answer": "..."}, ... ]```\n\n""" + f"The content is: {content}"

GPT = 4 # 3 is for GPT-3.5, 4 is GPT-4
if GPT == 3:
    MODEL = "gpt-3.5-turbo-1106"
    FILENAME = "gpt-3.5-data-qa.csv"
    PROMPT_COST = 0.001
    COMPLETION_COST = 0.002
elif GPT == 4:
    MODEL = "gpt-4-1106-preview"
    FILENAME = "gpt-4-data-qa.csv"
    PROMPT_COST = 0.01
    COMPLETION_COST = 0.03

SAVE_LOCATION = "data/"         # Location to save the data
N_CONCURRENT = 64               # Number of concurrent requests
MAX_RETRIES = 3                 # Maximum number of retries
BASE_WAIT_TIME = 2              # Base wait time in seconds
URL_MESSAGES = [(url, [         # List of tuples of (url, messages) to send through the API
    {"role": "system", "content": system},
    {"role": "user", "content": prompt(content)}
]) for url, content in zip(data["url"], data["data"])]

start_time = time.time()

async def async_openai_chat_call(client, messages, url, model, retries=0):
    "Async function to make a single OpenAI Chat API call"
    async_start_time = time.time()
    try:
        response = await client.post(
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                     'Content-Type': 'application/json'},
            json={'model': model, 'messages': messages, 'temperature': 0}
        )
        response.raise_for_status()  # Raise an exception for HTTP error codes
        response = response.json()
        duration = time.time() - async_start_time
        print(f"Success! Complete in {duration:.2f}s with {response['usage']['total_tokens']} tokens for {url}")
        return json.dumps(response)
    except Exception as e:
        # Try again if we haven't reached the maximum number of retries
        if retries < MAX_RETRIES:
            wait_time = BASE_WAIT_TIME * (2 ** retries)  # Exponential backoff
            print(f"Error for {url}: {e}. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            return await async_openai_chat_call(client, messages, url, model, retries + 1)
        else:
            print(f"Failed after {MAX_RETRIES} retries for {url}: {e}")
            return None  # Or handle the failure in some other way

# Inspiration: https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio
async def gather_with_limit(n, *coros):
    "Async function to limit the number of concurrent coroutines"
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros))

async def make_concurrent_chat_calls(messages_list, n_concurrent, model):
    "Async function to make concurrent OpenAI Chat API calls"
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [async_openai_chat_call(client, messages, url, model) for url, messages in messages_list]
        responses = await gather_with_limit(n_concurrent, *tasks)
        return responses
    
async def main():
    "Main async function to make concurrent OpenAI Chat API calls"
    responses = await make_concurrent_chat_calls(
        messages_list=URL_MESSAGES, 
        n_concurrent=N_CONCURRENT, 
        model=MODEL
    )
    # Process responses...
    data['api_res'] = responses

    try: 
        # This parses the API outputs and gets the actual JSON output
        api_responses = [json.loads(api_res)['choices'][0]['message']['content'] for api_res in data['api_res']]
        data['questions'] = [json.dumps(json.loads(api_output.strip()[7:-3])) for api_output in api_responses]
    except Exception as e:
        print(f"Error parsing API JSON responses: {e}")
        pass

    try:
        # Get cost for the run using the number of tokens used
        total_prompt_tokens = sum([json.loads(api_res)['usage']['prompt_tokens'] for api_res in data['api_res']])
        total_completion_tokens = sum([json.loads(api_res)['usage']['completion_tokens'] for api_res in data['api_res']])

        total_cost = PROMPT_COST * (total_prompt_tokens/1000) + COMPLETION_COST * (total_completion_tokens/1000)
        print(f"\nTotal cost: ${total_cost:.4f}")
    except Exception as e:
        print(f"Error getting cost: {e}")
        pass

    # Save the data
    data.to_csv(SAVE_LOCATION+FILENAME, index=False)
    print(f"Saved data to {SAVE_LOCATION+FILENAME}!")

asyncio.run(main())

print("\n--- %s seconds ---" % (time.time() - start_time))