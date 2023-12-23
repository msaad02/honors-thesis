"""
Creating the question/answer dataset from the cleaned data made in 2_cleaning.py

In this script you have the choice of using GPT-3.5 or GPT-4. GPT-3.5 is much
faster, but GPT-4 is much more accurate. I recommend using GPT-3.5 for testing
and GPT-4 for final results.

This script parallizes 64 requests at a time using asyncronous functions. We 
find that a rate limit of 64 requests at a time is around our allowed limit, 
but this can change depending on prompt used, rate limit, etc. This could 
much more easily be done with a for loop, but this is significantly faster. 
Who likes to wait? See https://platform.openai.com/docs/guides/rate-limits
"""

import pandas as pd
import os
import time
import json
import asyncio
import httpx
import time

data = pd.read_csv("data/website_data.csv")

system = """You are a helpful question maker for SUNY Brockport. Given some information about the school, your job is to parse that information and generate realistic questions a prospective student or faculty member might ask. For instance, "How can I apply to financial aid?" is a question a student could reasonably ask. Generate up to five of such questions and respond to those questions in JSON format, where "question" is one field, and "answer" is another. In your answers be enthusiastic about the school and try to be as helpful as possible."""

# If changing prompt I highly recommend keeping the JSON specification the same. It makes parsing much easier, 
# ESPECIALLY for GPT-3.5. From my testing, GPT-4 usually outputs this format regardless.
prompt = lambda content: """Based on the content given generate questions. Keep your tone optimistic and helpful. Think about your answer carefully before responding, and be sure to answer in JSON format starting with \n```json\n[\n{\n    "question": "...",\n"answer": "..."}, ... ]```\n\n""" + f"The content is: {content}"

GPT = 4 # 3 is for GPT-3.5, 4 is GPT-4
if GPT == 3:
    MODEL = "gpt-3.5-turbo-1106"
    FILENAME = "gpt_3.5_data_qa.csv"
    PROMPT_COST = 0.001                     # Price per 1000 prompt tokens for GPT-3.5-turbo-1106 as of 12/22/2023
    COMPLETION_COST = 0.002                 # Price per 1000 completion tokens for GPT-3.5-turbo-1106 as of 12/22/2023
    N_CONCURRENT = 14                       # My limit is 160_000 token/min, so 14 concurrent requests works well
elif GPT == 4:
    MODEL = "gpt-4-1106-preview"
    FILENAME = "gpt_4_data_qa.csv"
    PROMPT_COST = 0.01                      # Price per 1000 prompt tokens for GPT-4-1106-preview as of 12/22/2023
    COMPLETION_COST = 0.03                  # Price per 1000 completion tokens for GPT-4-1106-preview as of 12/22/2023
    N_CONCURRENT = 20                       # My limit is 300_000 token/min limit (+ it's slower), so we can do 25 concurrent requests

SAVE_LOCATION = "data/"         # Location to save the data
MAX_RETRIES = 3                 # Maximum number of retries
BASE_WAIT_TIME = 6              # Base wait time in seconds

# data = data.sample(n=30)      # Sample the data (for testing)

URL_MESSAGES = [(url, [         # List of tuples of (url, messages) to send through the API
    {"role": "system", "content": system},
    {"role": "user", "content": prompt(content)}
]) for url, content in zip(data["url"], data["data"])]

start_time = time.time()

async def async_openai_chat_call(client: httpx.AsyncClient, messages, url, model, retries=0):
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
    
def parse_output_for_questions(api_res):
    "Function to parse the API output for questions"
    try:
        response = json.loads(api_res)['choices'][0]['message']['content']
        # Remomove first 7 and last 3 (they are "```json" and "```" respectively), then standardize the jsons
        questions = json.dumps(json.loads(response.strip()[7:-3]))
    except:
        # If there is an error (hallucination most likely), return None
        questions = None
    return questions

async def main():
    "Main async function to make concurrent OpenAI Chat API calls"
    responses = await make_concurrent_chat_calls(
        messages_list=URL_MESSAGES, 
        n_concurrent=N_CONCURRENT, 
        model=MODEL
    )
    # Process responses...
    data['api_res'] = responses

    # If *somehow* the following crashes, the raw API responses are saved to a csv. This stuff gets expensive!
    # However, everything is wrapped in try/except blocks so it should be fine.
    data.to_csv(SAVE_LOCATION+"TEMP_GPT_GENERATED_DATA.csv", index=False)

    # This parses the API outputs and gets the actual JSON output
    data['questions'] = data['api_res'].apply(parse_output_for_questions)

    try:
        # Get cost for the run using the number of tokens used
        not_null_api_responses = data[data['api_res'].notnull()]['api_res']
        total_prompt_tokens = sum([json.loads(api_res)['usage']['prompt_tokens'] for api_res in not_null_api_responses])
        total_completion_tokens = sum([json.loads(api_res)['usage']['completion_tokens'] for api_res in not_null_api_responses])

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