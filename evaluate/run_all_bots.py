"""
The goal of evaluate.py is to provide some metrics to "grade" each of the models.

The working idea for this script is first to be able to ask a bunch of questions
from a json file for each of the models to answer. Results should be stored in a 
json file named "answer_set.json". In evaluate.py we will evaluate these with GPT-4.
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

model_list = {
    "Categorized": QuestionAnswering(verbose=False),
    "Traditional": TraditionalRAGEngine(),
    "Finetuned": FineTunedEngine(model_type=("gptq" if is_available() else "gguf")),
    "Scratch": ScratchModelEngine()
}

def run_all_models(question: str, models: dict) -> dict:
    # Error out if type mismatch.
    assert(isinstance(question, str))
    assert(isinstance(models, dict))

    answers = {}
    for name, model in tqdm(models.items()):
        answers[name] = model(question)

    return answers

def main():
    # Read in the questions to ask.
    with open('/home/msaad/workspace/honors-thesis/evaluate/eval_questions.json') as f:
        data = json.load(f)

    answer_set = {}
    for qa_pair in data:
        question, eval_info = qa_pair['question'], qa_pair['eval_info']
        
        print("Question:", question)
        answers = run_all_models(question=question, models=model_list)
        answers['Eval_Info'] = eval_info

        answer_set[question] = answers

        print("-----------")
        print(json.dumps(answers, indent=4))
        print("-----------")

    # Write the answer set to a json file.
    with open('/home/msaad/workspace/honors-thesis/evaluate/answer_set.json', 'w') as f:
        json.dump(answer_set, f, indent=4)
    
    print("Finished!")


if __name__ == "__main__":
    main()