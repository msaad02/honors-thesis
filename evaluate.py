from chatgpt_pe.categorized_engine import QuestionAnswering
import os


qa_bot = QuestionAnswering()

answer, price = qa_bot("How can I apply to SUNY Brockport?")

print(answer)