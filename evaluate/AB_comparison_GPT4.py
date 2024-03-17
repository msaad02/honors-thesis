"""
This script contains a class that will oversee comparing the different models
via GPT-4 results. There is no UI here, it randomly picks model arguments and
puts them side by side. Best model wins. Results are stored in a .csv file.
"""

# Imports
from fine_tuning.finetune_class import FineTunedEngine
from scratch_model.inference import ScratchModel
from text_search.rag import RAG
from torch.cuda import is_available
from termcolor import colored
from typing import Any, Optional
from tensorflow.python.keras.backend import clear_session


class compareModelsWithGPT4():
    def __init__(self):
        pass