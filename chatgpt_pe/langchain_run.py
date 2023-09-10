"""
Script to initiate a terminal chat interface using langchain and chromadb.
It utilizes semantic search with Chroma and leverages the GPT-3.5 model 
from the ChatOpenAI to answer questions. The interaction history is buffered
to provide context-aware responses.

Ensure you're familiar with the specifics in Version 2 of the README.md 
and refer to https://python.langchain.com/docs/get_started/introduction.html
for comprehensive documentation on the modules used.
"""

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Directory path for persisting vector data
vectordb_persist_dir = "/home/msaad/workspace/honors-thesis/data-collection/data/chroma_persist_dir"

# Very nice and relevant docs
# https://python.langchain.com/docs/modules/chains/popular/chat_vector_db

# Configuration for semantic search
search_args = {
    "score_threshold": .8,
    "k": 3
}

embedding_function = HuggingFaceBgeEmbeddings(
    model_name = "BAAI/bge-small-en",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True}
)

# Initialize retriever using Chroma with OpenAI embeddings for semantic search
retriever = Chroma(
    persist_directory = vectordb_persist_dir, 
    embedding_function = embedding_function
).as_retriever(search_kwargs = search_args)

# Set up memory buffer to store chat history for context-aware responses
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

# Combine ChatOpenAI model with retriever to form a conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(), 
    retriever,
    memory=memory
)

# Welcome message to initiate the chat
print("Welcome to Brockport GPT! Please ask a question.")

# Continuous loop to get user input and provide answers
while True:
    question = input("\n>>> ")

    # Obtain answer using the conversational retrieval chain
    result = qa({"question": question}) 

    print(result['answer'])