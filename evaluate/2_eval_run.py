"""
I am running into Errors when running the full evaluation in `2_eval_run.py`.

Basically, the Scratch model decides at some point when it's evaluating
questions to just explode in memory. When this happens, evaluation moves
at a standstill, and the RAM/SSD usage goes up through the roof (this is
especially not healthy for my SSD). So, instead of that, this will try
to run the thing in batches. Despite the issues, the scratch model does
seem to work fine most of the time, it just eventually decides to explode.
Maybe if we run ~100 questions at a time, it won't explode. That is the
goal of this script. Also, the output wil be saved itermitently so that
it can be restarted if it crashes.
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # prevent tensorflow logs

# Set path to parent directory so we can import from other folders.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fine_tuning.finetune_class import FineTunedModel
from scratch_model.inference import ScratchModel
from text_search.rag import RAG
from torch.cuda import empty_cache
from tqdm import tqdm
import pandas as pd
import time

start_time = time.time()


# ----------------------------------------------------------------------------------------------------
# Pull in evaluation data
data = pd.read_csv('./data/evaluation_data.csv')
questions = data["question"].to_list()


# ----------------------------------------------------------------------------------------------------
# RAG
print("Beginning RAG evaluation...\n")
rag_start_time = time.time()

model = RAG(
    main_categorization_model_dir="../text_search/models/main_category",
    subcategorization_model_dir="../text_search/models/subcategory_models/",
    embeddings_file="../text_search/data/embeddings.pickle"
)

answers = []
for question in tqdm(questions):
    try:
        answer = model(
            question=question,
            config={
                "search_type": "typesense",
                "use_classifier": True,
                "n_results": 5,
                "model_kwargs": {"temperature": 0.0},
            }
        )
        answers.append(answer)
    except:
        answers.append("Error")

df = pd.DataFrame({"question": questions, "answer": answers})
df.to_csv("data/overall_rag_evaluation_answers.csv", index=False)

del model
empty_cache()

print("\nRAG: --- %s seconds ---" % (time.time() - rag_start_time))


# ----------------------------------------------------------------------------------------------------
# Finetuned 
print("\n\nBeginning FineTuned evaluation...")
finetuned_start_time = time.time()

model = FineTunedModel(repo_id="msaad02/BrockportGPT-7b")
print() # Space after loading model progress bar

answers = []
for question in tqdm(questions):
    try:
        answers.append(model(question, temperature=0.01))
    except:
        answers.append("Error")

df = pd.DataFrame({"question": questions, "answer": answers})
df.to_csv("data/overall_finetuned_evaluation_answers.csv", index=False)

del model
empty_cache()

print("\nFinetuned: --- %s seconds ---" % (time.time() - finetuned_start_time))

# ----------------------------------------------------------------------------------------------------
# Scratch
print("\n\nBeginning Scratch evaluation...")
scratch_start_time = time.time()

# Breaking things up into chunks to prevent memory errors
n = 5
questions_split = [questions[i * n:(i + 1) * n] for i in range((len(questions) + n - 1) // n )]

answers_total = []
with tqdm(total=len(questions)) as pbar:
    for id, questions_chunk in enumerate(questions_split):
        model = ScratchModel(model_dir="../scratch_model/models/transformer_v7/")

        tqdm.write(f"Running chunk {id + 1} of {len(questions_split)}")
                   
        answers_chunk = []
        for question in questions_chunk:
            try:
                answer = model(question, max_tokens=100)

                answers_chunk.append(answer)
                answers_total.append(answer)
            except:
                answers_chunk.append("Error")
                answers_total.append("Error")
            pbar.update()

        df = pd.DataFrame({"question": questions_chunk, "answer": answers_chunk})
        df.to_csv(f"data/scratch_parts/overall_scratch_evaluation_answers_{id}.csv", index=False)

        del model

if len(answers_total) != len(questions):
    raise ValueError("Number of answers does not match number of questions")
else:
    df = pd.DataFrame({"question": questions, "answer": answers_total})
    df.to_csv("data/overall_scratch_evaluation_answers.csv", index=False)

print("\nScratch: --- %s seconds ---" % (time.time() - scratch_start_time))

# ----------------------------------------------------------------------------------------------------
print("\n\n--- Total time: %s seconds ---" % (time.time() - start_time))