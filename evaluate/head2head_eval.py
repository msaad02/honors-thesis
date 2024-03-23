"""
This script is all about evaluating the models. We previously ran the models
on the test dataset in `./run_models.py` and now we are going to put the models
in a head-to-head competition to see which one is the best. The evaluation is 
scored by GPT-4, who will be the judge of the best response to a question.

Using these results, we can create some leaderboard of what the best model is.
"""

from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import random

client = OpenAI()

# Dataset
df = pd.read_csv("./data/test_answers.csv")

# Splitting the data into their various combinations
rag_v_scratch = df.loc[:, ['question', 'answer', 'RAG', 'Scratch']]
rag_v_ft = df.loc[:, ['question', 'answer', 'RAG', 'Finetuned']]
scratch_v_ft = df.loc[:, ['question', 'answer', 'Scratch', 'Finetuned']]

# Combining the data
dfs = [rag_v_scratch, rag_v_ft, scratch_v_ft]
result = pd.concat(dfs, ignore_index=True)

output = []

# Randomly select which model is player A and which is player B
for row in result.iterrows():
    row = row[1][~row[1].isna()]

    idx_a = random.choice([0, 1])
    idx_b = 1 - idx_a

    output.append({
        'question': row['question'],
        'answer': row['answer'],
        'player_a': row[idx_a+2],
        'player_b': row[idx_b+2],
        'whois_player_a': row.index[idx_a+2],
        'whois_player_b': row.index[idx_b+2]
    })

# Shuffle the data
df = pd.DataFrame(output).sample(frac=1).reset_index(drop=True)

# Create the prompt
prompt = lambda a,b,c,d: f"Question: {a}\nGround Truth: {b}\nPlayer A: {c}\nPlayer B: {d}"

# Create the prompt for each row
df['prompt'] = df.apply(lambda x: prompt(x['question'], x['answer'], x['player_a'], x['player_b']), axis=1)


# Create a function to assess the players
prompt_tokens = 0
completion_tokens = 0

def assess_players(prompt):
    "GPT-4 evaluates best model. Returns A, B, or None to indicate best response."
    global prompt_tokens, completion_tokens

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful referee, who helps pick the best response to a question. The question is about SUNY Brockport, a school in upstate NY. You are given the following:\n\n1) The question given.\n2) The ground truth in the form of an answer to the question.\n3) Player A response to the question.\n4) Player B response to the question.\n\nGiven the question and ground truth, select which player has the best response. Respond with either \"A\", or \"B\" only. In some cases, it may be possible that both players are incorrect. In those cases, respond with \"None\". In choosing the best response prioritize correctness first, then enthusiasm and overall coherence after. Remember to only respond with either \"A\", \"B\", or \"None\". Do not explain your decision."
                }, {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=1
        )
    except:
        return "Error"

    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens

    return response.choices[0].message.content


# Assess the players
for prompt in tqdm(df.loc[:, 'prompt']):
    df.loc[df['prompt'] == prompt, 'best_response'] = assess_players(prompt)

# Save the results
df.to_csv("data/test_evaluation.csv", index=False)

# Print the results
print("Done! Results saved to data/test_evaluation.csv.")

print(f"Prompt tokens used: {prompt_tokens}")
print(f"Completion tokens used: {completion_tokens}")