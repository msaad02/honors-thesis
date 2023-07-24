# Prompt Engineering with ChatGPT

One way to deal with ChatGPT's limited knowledge post-2021 is to inject additional data to your prompt. In doing so, ChatGPT can interpret it in different and creative ways. 

In this implementation we use semantic search via operand.ai. With operand, we are easily able to upload documents and websites for their service to then parse it. Eventually when we make a request, Operand returns the results according to semantic search. Once we have that output, we can finally call openai's ChatGPT api with our newfound data.




# IDEA:

Could do a keyword search or just an additional layer of semantic search to filter down into a category, where you could them semantic search text exclusive to that category. Things get messy when adding a ton of webpages to 1 search, this could allow for a much cleaner implementation that returns better results