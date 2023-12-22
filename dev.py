"""
Async requests with asyncio

Inspiration: https://stackoverflow.com/questions/48483348/how-to-limit-concurrency-with-python-asyncio
"""

import os
import httpx
import asyncio
import time

MODEL = "gpt-4-1106-preview"
N_CONCURRENT = 64
URL_MESSAGES = [("URL1", [
    {"role": "user", "content": "Respond with JSON! First test message"},
    {"role": "user", "content": "Second test message"}
]), ("URL2", [
    {"role": "user", "content": "Respond with JSON! First test asdf"},
    {"role": "user", "content": "Second test message"}
]), ("URL3", [
    {"role": "user", "content": "Respond with JSON! First test message"},
    {"role": "user", "content": "Second test zxcvzxcv"}
]), ("URL4", [
    {"role": "user", "content": "Respond with JSON! First test asdfasdf"},
    {"role": "user", "content": "Second test message"}
])]

start_time = time.time()

async def async_openai_chat_call(client, messages, url, model):
    async_start_time = time.time()
    response = await client.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
            'Content-Type': 'application/json'
        },
        json={
            'model': model,
            'messages': messages
        }
    )
    response = response.json()
    duration = time.time() - async_start_time

    print(f"Success! Complete in {duration:.2f}s with {response['usage']['total_tokens']} tokens for {url}")

    return response

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