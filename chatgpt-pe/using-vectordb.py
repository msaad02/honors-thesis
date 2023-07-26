
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

vectordb_persist_dir = "/home/msaad/workspace/honors-thesis/data-collection/data/chroma_persist_dir"

# Very nice and relevant docs
# https://python.langchain.com/docs/modules/chains/popular/chat_vector_db

retreiver = Chroma(
    persist_directory = vectordb_persist_dir, 
    embedding_function = OpenAIEmbeddings()
).as_retriever()

# qa = RetrievalQA.from_chain_type(
#     llm=OpenAI(), 
#     chain_type="map_reduce", 
#     retriever=retreiver
# )

vectordbkwargs = {"search_distance": 0.9}
# retreiver.search_kwargs = 

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# qa = ConversationalRetrievalChain.from_llm(
#     OpenAI(), 
#     retreiver, 
#     memory=memory,
#     return_source_documents=True
# )

chat_history = []

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retreiver, return_source_documents=True, memory=memory)
chat_history = []


# I wonder how I can implement search_distance and memory simultaneously. Might work best.
# Current error with that : ValueError: One input key expected got ['vectordbkwargs', 'question']

print("Welcome to Brockport GPT! Please ask a question.")
while True:
    question = input("\n>>> ")

    # result = qa.run(question)
    result = qa({"question": question, "chat_history": chat_history, "vectordbkwargs": vectordbkwargs})

    print(result['answer'])