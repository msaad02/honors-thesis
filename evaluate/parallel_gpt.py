"""
Running evaluations takes forever. This script is used to run evaluations
in parallel to speed up the process. We are NOT using the official OpenAI
library for this, rather just using httpx to make the API calls. This is
because the implementation I had in data_collection does not use the OpenAI
library, and I'm just reworking that code here.
"""

import pandas as pd
import os
import time
import json
import asyncio
import httpx
import time
from tqdm.asyncio import tqdm_asyncio

class ParallelGPT():
    def __init__(
        self,
        gpt: int = 3,
        max_retries: int = 2,
        base_wait_time: int = 4
    ):
        self.gpt = gpt
        self.max_retries = max_retries
        self.base_wait_time = base_wait_time

        if self.gpt == 3:
            self.model = "gpt-3.5-turbo"
            self.prompt_cost = 0.001                     # Price per 1000 prompt tokens for GPT-3.5-turbo-1106 as of 12/22/2023
            self.completion_cost = 0.002                 # Price per 1000 completion tokens for GPT-3.5-turbo-1106 as of 12/22/2023
            self.n_concurrent = 14                       # My limit is 160_000 token/min, so 14 concurrent requests works well
        elif self.gpt == 4:
            self.model = "gpt-4-turbo-preview"
            self.prompt_cost = 0.01                      # Price per 1000 prompt tokens for GPT-4-1106-preview as of 12/22/2023
            self.completion_cost = 0.03                  # Price per 1000 completion tokens for GPT-4-1106-preview as of 12/22/2023
            self.n_concurrent = 20                       # My limit is 300_000 token/min limit, so 20 concurrent requests works well
        else:
            raise ValueError("GPT must be 3 or 4")
        
    async def async_openai_chat_call(
        self,
        client: httpx.AsyncClient, 
        prompt: list[dict[str, str], dict[str, str]], 
        model_params: dict,
        retries: int = 0,
    ):
        "Async function to make a single OpenAI Chat API call"
        async_start_time = time.time()
        try:
            response = await client.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                        'Content-Type': 'application/json'},
                json={'model': self.model, 'messages': prompt, **model_params}
            )
            response.raise_for_status()  # Raise an exception for HTTP error codes
            response = response.json()
            duration = time.time() - async_start_time
            tqdm_asyncio.write(f"Success! Complete in {duration:.2f}s with {response['usage']['total_tokens']} tokens")
            return json.dumps(response)
        except Exception as e:
            # Try again if we haven't reached the maximum number of retries
            try:
                if retries < self.max_retries:
                    wait_time = self.base_wait_time * (2 ** retries)  # Exponential backoff
                    tqdm_asyncio.write(f"Error: {e}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    return await self.async_openai_chat_call(client, prompt, model_params, retries + 1)
                else:
                    tqdm_asyncio.write(f"Failed after {self.max_retries} retries: {e}")
                    return None  # Or handle the failure in some other way
            except:
                return None

    # Inspiration: https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio
    async def gather_with_limit(self, n, *coros):
        "Async function to limit the number of concurrent coroutines"
        semaphore = asyncio.Semaphore(n)

        async def sem_coro(coro):
            async with semaphore:
                return await coro
            # replaced asyncio with tqdm_asyncio
        return await tqdm_asyncio.gather(*(sem_coro(c) for c in coros))


    async def make_concurrent_chat_calls(self, prompt_list, n_concurrent, model_params):
        "Async function to make concurrent OpenAI Chat API calls"
        async with httpx.AsyncClient(timeout=None) as client:
            tasks = [
                self.async_openai_chat_call(client=client, prompt=prompt, model_params=model_params) for prompt in prompt_list
            ]
            responses = await self.gather_with_limit(n_concurrent, *tasks)
            return responses
        

    def parse_output_for_questions(sefl, api_res):
        "Function to parse the API output for questions"
        try:
            response = json.loads(api_res)['choices'][0]['message']['content']
        except:
            print("Error parsing output")
            response = "ERROR" # If there is an error (hallucination most likely), return None
        return response
        

    async def parallel_gpt(
        self,
        data: pd.Series,
        model_params: dict = {}
    ):
        start_time = time.time()
        "Main async function to make concurrent OpenAI Chat API calls"
        responses = await self.make_concurrent_chat_calls(
            prompt_list=data, 
            n_concurrent=self.n_concurrent,
            model_params=model_params
        )
    
        df = pd.DataFrame({
            "prompt": data,
            "api_res": responses
        })

        try:
            df["output"] = df["api_res"].apply(self.parse_output_for_questions)
        except:
            print("Error parsing output")
            df.to_csv("TMP_ERROR_OUT_FILE.csv", index=False)
            return None

        try:
            # Get cost for the run using the number of tokens used
            not_null_api_responses = df[~df['api_res'].isna()]['api_res']
            total_prompt_tokens = sum([json.loads(api_res)['usage']['prompt_tokens'] for api_res in not_null_api_responses])
            total_completion_tokens = sum([json.loads(api_res)['usage']['completion_tokens'] for api_res in not_null_api_responses])

            total_cost = self.prompt_cost * (total_prompt_tokens/1000) + self.completion_cost * (total_completion_tokens/1000)
            print(f"\nTotal cost: ${total_cost:.4f}")
        except Exception as e:
            print(f"Error getting cost: {e}")
            pass

        print("\n--- %s seconds ---" % (time.time() - start_time))

        return df['output']
