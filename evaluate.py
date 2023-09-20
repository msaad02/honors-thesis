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
from chatgpt_pe.categorized_engine import QuestionAnswering
from chatgpt_pe.rag_engine import TraditionalRAGEngine
from fine_tuning.finetune_engine import FineTunedEngine
from scratch_model.scratch_model_engine import ScratchModelEngine

