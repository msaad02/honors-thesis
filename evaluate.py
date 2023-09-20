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
import json
from tqdm import tqdm
from chatgpt_pe.categorized_engine import QuestionAnswering
from chatgpt_pe.rag_engine import TraditionalRAGEngine
from fine_tuning.finetune_engine import FineTunedEngine
from scratch_model.scratch_model_engine import ScratchModelEngine

question = "How can I apply to SUNY Brockport?"

model_list = {
    "Categorized": QuestionAnswering(verbose=False),
    "Traditional": TraditionalRAGEngine(),
    "Finetuned": FineTunedEngine(model_type="gguf"),
    "Scratch": ScratchModelEngine()
}

# Begin this developing a JSON file to store relevant information here.
# Importantly, we also want metadata collected (parameters for model).
# (There shouldn't be many parameters honestly -- but seeding could be good)

def run_all_models(question: str, models: dict):
    # Error out if type mismatch.
    assert(isinstance(question, str))
    assert(isinstance(models, dict))
    # Probably good idea to make sure contents of model is type of class from my classes but... too much work.

    answers = {}
    for name, model in tqdm(models.items()):
        answers[name] = model(question)

    return answers


answers = run_all_models(question=question, models=model_list)

print(answers)

# Answers
# {
#     'Categorized': 'To apply to SUNY Brockport, you need to either use the Common App or the SUNY App. Additionally, you will need to provide your Official High School transcript and at least one letter of recommendation.', 
#     'Traditional': ' You can apply to SUNY Brockport by visiting their website and submitting an online application.', 
#     'Finetuned': 'You can apply to SUNy Brockport by following the application process outlined on their website.', 
#     'Scratch': 'You can apply to Suny Brockport by visiting their website and filling out the application.'
# }
