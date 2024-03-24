# RAG Evaluation

Using the finetuned and scratch model are straightforward since there are
no additional configurations needed. However, the RAG model requires a specific
runtime information such as the search type, whether the use the classfier, and
the number of results to return. This information is passed in the config dictionary.

However, running RAG against all the other models to determine which the best model
in would cause an explosion of data. Instead, what we want to do is just compare the
best RAG configuration against the other models.

However, we still need to figure out what the best configuration is. The goal of the
RAG evaluation process is to determine this configuration. We will do this by running
a similar head to head comparison between the different RAG configurations to determine
which is the best.