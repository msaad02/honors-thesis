"""
Running the head-to-head comparison of the RAG evaluation data.

GPT-4 is the evaluator, this functions very similarily to `../3_eval_comparison.py`
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # prevent tensorflow logs

# Set path to parent directory so we can import from other folders.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evaluate.parallel_gpt import ParallelGPT
import pandas as pd
import asyncio

parallel_gpt = ParallelGPT(gpt=4)

message = lambda a, b, c, d: [
    {"role": "system", "content": """You are a helpful referee who helps pick the best response to a question. The question is about SUNY Brockport, a school in upstate NY. You are given the following:\n\n1) The question given.\n2) The ground truth in the form of an answer to the question.\n3) Player A response to the question.\n4) Player B response to the question.\n\nGiven the question and ground truth, select which player has the best response. Respond with either \"A\", or \"B\" only. In some cases, it may be possible that both players are incorrect. In those cases, respond with \"None\". In choosing the best response prioritize correctness first, then enthusiasm and overall coherence after. Remember to only respond with either \"A\", \"B\", or \"None\". Do not explain your decision."""},
    {"role": "user", "content": f"Question: {a}\nGround Truth: {b}\nPlayer A: {c}\nPlayer B: {d}"}
]

df = pd.read_csv("../data/rag_evaluation_output.csv")

# Create the prompt for each row
df['prompt'] = df.apply(lambda x: message(x['question'], x['answer'], x['player_A_response'], x['player_B_response']), axis=1)


df['best_response'] = asyncio.run(parallel_gpt.parallel_gpt(
    data=df['prompt'],
    model_params={
        'temperature': 0,
        'max_tokens': 1
    }
))

df.to_csv("../data/rag_evaluation_comparison.csv", index=False)

print("\nDone!\n")