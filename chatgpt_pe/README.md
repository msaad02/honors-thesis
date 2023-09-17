# Prompt Engineering/Retreival Augmented Generation (RAG)

One way to deal with Large Language Models (LLMs) limited knowledge post-2021, or limited specific domain knowledge is to inject additional information to your prompt. In doing so, the LLM is able to take and interpret that additional information to generate a response in context of current day information. This is the idea behind Prompt Engineering/Retreival Augmented Generation (RAG), and is the basis for this portion of the project. In this portion of the project, we will be using semantic search to augment our LLMs so they can better answer questions.

## Technical Steps

### Version 3 (Best performing)

This version of the implementation uses a "categorized engine", as I'm calling it. Since I scraped all this data for this project off the web, I have the URL and the raw data. Having this information, I can use the filestructure built in to the URL to categorize the data. For example, if I have a URL like **brockport.edu/academics/computing-sciences/**, I can use the filestructure to categorize the data as **academics** and then **computing-sciences**. This allows me to easily query the data based on category, and then semantic search the results of that query with more relevant results. To do this, I'm leveraging GPT-4 and careful prompting to both categorize and answer the question. This is the best performing of the 3 versions, but its downside is that it also costs the most per run by quite a margin (~$0.03/question).

### Version 2

The current version of this version uses langchain and chromadb. These allow for many of the same benefits of Operand alongside the added benefit of running locally, as opposed to it being powered by some external service. However, alongside this benefit, running things locally means I need to handle the data -- something that Operand handled prior. My dataset, coming from raw HTMLs, required cleaning and then chunking (for semantic search). These steps happen in the script [data_cleaning_for_ss.py](../data_collection/data_cleaning_for_ss.py), refer there for more details.

Like the old implementation, this version uses GPT3.5 (ChatGPT) from openai as the LLM behind the scenes.

### Version 1 (Old in ./dev/)

In this the Version 1 implementation we use semantic search via operand.ai. With operand, we are easily able to upload documents and websites for them to parse. From there, you can query your documents based on similarity with semantic search. Once we have the output, we can pass in the additional context to our LLM for interpretation.

This implementation utilizes GPT3.5 from openai as the LLM powering this chatbot.

> NOTE: Shortly after implementing Version 1, operand.ai shutdown their service. This trigged development of Version 2
