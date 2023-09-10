"""
What is the goal of evaluate.py?

Still in question... But, for now, I want to be able to run all the models and see which one is the best.

Basically just like a side by side comparison for them all
"""
from chatgpt_pe.categorized_engine import QuestionAnswering
import os


qa_bot = QuestionAnswering()

answer, price = qa_bot("How can I apply to SUNY Brockport?")

print(answer)