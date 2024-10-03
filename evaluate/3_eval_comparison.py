"""
This script is all about evaluating the models. We previously ran the models
on the eval dataset in `./eval_run.py` and now we are going to put the models
in a head-to-head competition to see which one is the best. The evaluation is 
scored by GPT-4, who will be the judge of the best response to a question.

Using these results, we can create some leaderboard of what the best model is.
"""

import asyncio
from parallel_gpt import ParallelGPT
import pandas as pd
import tiktoken

# Select either GPT-3.5 or GPT-4
parallel_gpt = ParallelGPT(gpt=4)

df_rag = pd.read_csv("data/overall_rag_evaluation_answers.csv")
df_ft = pd.read_csv("data/overall_finetuned_evaluation_answers.csv")
df_scr = pd.read_csv("data/overall_scratch_evaluation_answers.csv")

df_rag['type'] = "RAG"
df_ft['type'] = "Finetuned"
df_scr['type'] = "Scratch"

df_rag = df_rag.drop_duplicates(subset="question").reset_index(drop=True)
df_ft = df_ft.drop_duplicates(subset="question").reset_index(drop=True)
df_scr = df_scr.drop_duplicates(subset="question").reset_index(drop=True)

def make_matchups(df1, df2):
    """Creates a head-to-head matchup dataframe between two dataframes."""
    assert len(df1) == len(df2), "Dataframes must have the same length."

    df_len = len(df1)

    df1_1, df1_2 = df1.iloc[:df_len//2], df1.iloc[df_len//2:]
    df2_1, df2_2 = df2.iloc[:df_len//2], df2.iloc[df_len//2:]

    df_1 = pd.merge(df1_1, df2_1, on="question", suffixes=["_A", "_B"])
    df_2 = pd.merge(df2_2, df1_2, on="question", suffixes=["_A", "_B"])

    df = pd.concat([df_1, df_2])
    return df

df_rag_ft = make_matchups(df_rag, df_ft)
df_rag_scr = make_matchups(df_rag, df_scr)
df_ft_scr = make_matchups(df_ft, df_scr)

df = pd.concat([df_rag_ft, df_rag_scr, df_ft_scr])

df_eval = pd.read_csv("data/evaluation_data.csv")
df_eval.rename({"answer": "true_answer", "type": "question_type"}, axis=1, inplace=True)

df_eval = df_eval.drop_duplicates(subset="question").reset_index(drop=True)

df = pd.merge(df, df_eval, on="question", how="left")

message = lambda a, b, c, d: [
    {"role": "system", "content": """You are a helpful referee who helps pick the best response to a question. The question is about SUNY Brockport, a school in upstate NY. You are given the following:\n\n1) The question given.\n2) The ground truth in the form of an answer to the question.\n3) Player A response to the question.\n4) Player B response to the question.\n\nGiven the question and ground truth, select which player has the best response. Respond with either \"A\", or \"B\" only. In some cases, it may be possible that both players are incorrect. In those cases, respond with \"None\". In choosing the best response prioritize correctness first, then enthusiasm and overall coherence after. Remember to only respond with either \"A\", \"B\", or \"None\". Do not explain your decision."""},
    {"role": "user", "content": f"Question: {a}\nGround Truth: {b}\nPlayer A: {c}\nPlayer B: {d}"}
]

df['prompt'] = df.apply(lambda x: message(x['question'], x['true_answer'], x['answer_A'], x['answer_B']), axis=1)
# df = df[['question', 'prompt', 'question_type', 'type_A', 'type_B']]

encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-4")

def num_tokens_from_string(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

prompt_inputs = [" ".join([d['content'] for d in df['prompt'][i]]) for i in range(len(df))]
prompt_tokens = [num_tokens_from_string(prompt, encoding) for prompt in prompt_inputs]

expected_cost = (sum(prompt_tokens) / 1_000_000) * 10 + (len(df) / 1_000_000) * 30
print(f"Expected cost: ${expected_cost:.2f}")


df['best_response'] = asyncio.run(parallel_gpt.parallel_gpt(
    data=df['prompt'],
    model_params={
        'temperature': 0,
        'max_tokens': 1
    }
))


df.to_csv("data/overall_evaluation.csv", index=False)

print("Evaluation complete! Results saved to data/overall_evaluation.csv")