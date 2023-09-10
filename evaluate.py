"""
What is the goal of evaluate.py?

Still in question... But, for now, I want to be able to run all the models and see which one is the best.

Basically just like a side by side comparison for them all
"""
from chatgpt_pe.categorized_engine import QuestionAnswering
from chatgpt_pe.rag_engine import TraditionalRAGEngine

question = "How can I apply to SUNY Brockport?"

categorized_qa = QuestionAnswering()
traditional_qa = TraditionalRAGEngine()

answer, price = categorized_qa(question)

print("Categorized QA:\n", answer)

answer = traditional_qa(question)

print("\n\nNoncategorized QA:\n", answer)