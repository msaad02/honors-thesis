# Prompt Engineering/Retreival Augmented Generation (RAG)

One way to deal with Large Language Models (LLMs) limited knowledge post-2021 is to inject additional current information to your prompt. In doing so, the LLM is able to take and interpret that additional information to generate a response in context of current day information.

## Technical Steps

### Version 2 (Current)

The current version of this version uses langchain and chromadb. These allow for many of the same benefits of Operand alongside the added benefit of running locally, as opposed to it being powered by some external service. However, alongside this benefit, running things locally means I need to handle the data -- something that Operand handled prior. My dataset, coming from raw HTMLs, required cleaning and then chunking (for semantic search). These steps happen in the script [data_cleaning_for_ss.py](../data-collection/data_cleaning_for_ss.py), refer there for more details.

Like the old implementation, this version uses GPT3.5 (ChatGPT) from openai as the LLM behind the scenes.

### Version 1

In this the Version 1 implementation we use semantic search via operand.ai. With operand, we are easily able to upload documents and websites for them to parse. From there, you can query your documents based on similarity with semantic search. Once we have the output, we can pass in the additional context to our LLM for interpretation.

This implementation utilizes GPT3.5 from openai as the LLM powering this chatbot.

> NOTE: Shortly after implementing Version 1, operand.ai shutdown their service. This trigged development of Version 2


# IDEAS FOR THE FUTURE:

1. Could do a keyword search or just an additional layer of semantic search to filter down into a category, where you could them semantic search text exclusive to that category. Things get messy when adding a ton of webpages to 1 search, this could allow for a much cleaner implementation that returns better results
2. Semantic search the question database instead of the raw html text.