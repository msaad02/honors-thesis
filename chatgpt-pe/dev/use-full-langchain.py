from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

vectordb_persist_dir = "/home/msaad/workspace/honors-thesis/data-collection/data/chroma_persist_dir"

# Very nice and relevant docs
# https://python.langchain.com/docs/modules/chains/popular/chat_vector_db

search_args = {
    "score_threshold": .8,
    "k": 3
}

retriever = Chroma(
    persist_directory = vectordb_persist_dir, 
    embedding_function = OpenAIEmbeddings()
).as_retriever(search_kwargs = search_args)

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(), 
    retriever,
    memory=memory
)

print("Welcome to Brockport GPT! Please ask a question.")
while True:
    question = input("\n>>> ")

    # result = qa.run(question)
    result = qa({"question": question}) 

    print(result['answer'])