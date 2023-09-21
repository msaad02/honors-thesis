"""
Idea:

GPT-4 evaluation has become a common use case as of late, with varying
degrees of success. The idea is to have GPT-4 evaluate the answer given
by the bot and compare it to the best answer. The best answer is the
answer that the bot should have given. The bot's answer is the answer
that the bot actually gave. The bot's answer is compared to the best
answer and given a score from 1-10. 1 being the worst and 10 being the
best. The score is then averaged over all the questions in the question
set. The average score is the score for the bot. The higher the score,
the better the bot is at answering questions.
"""
import json

with (open("./answer_set.json", "r")) as f:
    answer_set = json.load(f)


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