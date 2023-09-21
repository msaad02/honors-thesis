"""
The goal of evaluate.py is to provide some metrics to "grade" each of the models.

The working idea for this script is first to be able to ask a bunch of questions
from a txt file for each of the models to answer. Results should be stored in a 
json file named "answers.txt". Afterwards, we will consider grading techniques.

# Grading Ideas:
GPT-4 evaluation has become a common use case as of late, and it could also be
implemented in this project. For instance, if for the set of "master questions",
we also create a set of "master answers" with key information highlighted, we
could have GPT-4 evaluate whether or not the important information is inside the
answer as well as grammar, usefulness, and a magnitude of other things.
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # prevent tensorflow logs

# Set path to parent directory so we can import from other folders.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from tqdm import tqdm
from torch.cuda import is_available

from chatgpt_pe.categorized_engine import QuestionAnswering
from chatgpt_pe.rag_engine import TraditionalRAGEngine
from fine_tuning.finetune_engine import FineTunedEngine
from scratch_model.scratch_model_engine import ScratchModelEngine

question = "How can I apply to SUNY Brockport?"

model_list = {
    "Categorized": QuestionAnswering(verbose=False),
    "Traditional": TraditionalRAGEngine(),
    "Finetuned": FineTunedEngine(model_type=("gptq" if is_available() else "gguf")),
    "Scratch": ScratchModelEngine()
}

# Begin this developing a JSON file to store relevant information here.
# Importantly, we also want metadata collected (parameters for model).
# (There shouldn't be many parameters honestly -- but seeding could be good)

def run_all_models(question: str, models: dict):
    # Error out if type mismatch.
    assert(isinstance(question, str))
    assert(isinstance(models, dict))

    answers = {}
    for name, model in tqdm(models.items()):
        answers[name] = model(question)

    return answers

def eval_model(answer: str, best_answer: str):
    """
    Consider ways to evaluate questions.  1-10 scale?

    See "Grading ideas" section at the top in the docstring.

    Might consider adding another column to the quesiton db
    that contains the "MUST CONTAIN" info type column that
    GPT4 can easily evaluate whether its in the answer given.
    """
    assert(isinstance(answer, str))
    assert(isinstance(best_answer, str))

    return 10


def main():

    with open('/home/msaad/workspace/honors-thesis/evaluate/eval_questions.json') as f:
        data = json.load(f)

    answer_set = {}
    for qa_pair in data:
        question, answer = qa_pair['question'], qa_pair['answer']
        
        print("Question:", question)
        answers = run_all_models(question=question, models=model_list)

        answer_set[question] = answers

        print("-----------")
        print(json.dumps(answers, indent=4))
        print("-----------")

    # open a file for writing
    with open('/home/msaad/workspace/honors-thesis/evaluate/answer_set.json', 'w') as f:
        # write the dictionary to the file in JSON format
        json.dump(answer_set, f, indent=4)




if __name__ == "__main__":
    main()