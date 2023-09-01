from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
import openai
import json
from colorama import init as colorama_init
from colorama import Fore, Style
colorama_init()

# Base CHROMADB categorized data dirs
base_categorized_chroma_path = "/home/msaad/workspace/honors-thesis/data-collection/data/categorized_datastore/chroma_data/"

# Load in the JSON file
with open("/home/msaad/workspace/honors-thesis/data-collection/data/categorized_data.json", "r") as f:
    data = json.load(f)

# Load semantic search model
model = HuggingFaceBgeEmbeddings(
    model_name = "BAAI/bge-small-en",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
)


#########################################
# Helper Functions

def query_gpt(system: str, prompt: str, model_name: str = "gpt-4", temperature: int = 0, max_tokens: int = 10, price: float = 0):
    """
    Query one of the openai GPT family models for a given system/prompt pair, and return the response.
    """
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Calculate pricing as of 8/31/2023
    if model_name == "gpt-4":
        price = price + response['usage']['prompt_tokens'] * 0.03/1000 + response['usage']['completion_tokens'] * 0.06/1000
    elif model_name == "gpt-3.5-turbo":
        price = price + response['usage']['prompt_tokens'] * 0.0015/1000 + response['usage']['completion_tokens'] * 0.002/1000

    try:
        return eval(response.to_dict()['choices'][0]['message']['content']), price
    except:
        return response.to_dict()['choices'][0]['message']['content'], price


def get_relevant_docs_for_dir(path: str, question: str, model: HuggingFaceBgeEmbeddings = model):
    """
    Return relevant documents for a given question and category (directory).

    Incase this is not clear, every single categorized directory has their own independent Chromadb.
    This function queries the ChromaDB for a given directory, and returns the relevant documents for a given question.
    """

    # Configuration for semantic search
    search_args = {
        "score_threshold": .8,
        "k": 3
    }

    # Initialize retriever using Chroma with OpenAI embeddings for semantic search
    retriever = Chroma(
        persist_directory = path, 
        embedding_function = model
    ).as_retriever(search_kwargs = search_args)

    relevant_docs = retriever.get_relevant_documents(question)

    return relevant_docs




def answer_question_with_categorization(question: str) -> str:
    price = 0

    """
    This function handles the entire process of answering a question with the categorized search engine.
    
    Input a question, output an answer.
    """


    # FIRST PROMPT/INITIAL CATEGORIZATION
    initial_categorization_system = "You are a helpful classification system. Categorize a question into its category based on the brief description provided."
    initial_categorization_prompt = f"""\
    The question is: 
    {question}

    The following categories available are:
    "none": if the question does not fit into any of the above categories, or are not related to SUNY Brockport
    "live": for policy related questions
    "academics": academic related information to majors, or programs.
    "support": current student and faculty support
    "life": information about student life
    "about": information about the university, such as Title IX, mission statement, diversity, or strategic plan, local area, president, etc.
    "admissions": for prospective students looking to apply
    "graduate": information about graduate programs
    "admissions-aid": information about admissions and financial aid
    "scholarships-aid": information about scholarships and financial aid
    "library": information about the library
    "research-foundation": information about the research at Brockport

    Respond ONLY with the name of the category. (i.e. live, academics, etc.). If a question does not fit into any of the above categories, or is otherwise inappropriate, respond with "none"."""

    first_category, price = query_gpt(
        system=initial_categorization_system, 
        prompt=initial_categorization_prompt,
        temperature=0,
        price=price
    )

    print(f"\n{Fore.GREEN}FIRST CATEGORY: {Style.RESET_ALL}{first_category}")

    # Now lets get the list of subcategories for the first category
    subcategory_keys = data[first_category].keys() # NOTE: This includes URLs AND subcategories. Need to filter out URLs
    subcategories = [non_url for non_url in subcategory_keys if not non_url.startswith("http")] # Filters out URLs
    print_pretty_subcategories = "\n".join(subcategories)

    path_to_first_category = f"{base_categorized_chroma_path}{first_category}/"

    print(f"\n{Fore.GREEN}PATH TO FIRST CATEGORY: {Style.RESET_ALL}{path_to_first_category}")

    vector_search_results = get_relevant_docs_for_dir(question=question, path=path_to_first_category)
    print_pretty_vector_search_results = "\n".join([doc.page_content for doc in vector_search_results])

    print(f"\n{Fore.BLUE}VECTOR SEARCH RESULTS: \n{Style.RESET_ALL}{print_pretty_vector_search_results}")

    # FINISHED FIRST PROMPT/INITIAL CATEGORIZATION






    ###############################################################
    # SECOND PROMPT/SUBCATEGORIZATION

    subcategory_system = "You are a helpful assistant. Given a context and a question, you will either answer the question, or choose the most relevant category for the question."
    subcategory_prompt = f"""\
    The question is:
    {question}

    If the answer is available in the following data, answer the question. If not, choose one of the following subcategories.
    The decision is yours whether to search more, or take your current information.

    The current information is:
    {print_pretty_vector_search_results}

    The following subcategories available are:
    {print_pretty_subcategories}

    If you choose a subcategory, respond ONLY with the name of that category. (i.e. "live", "academics", etc.)."""

    #### BEGIN LOGIC AFTER SECOND PROMPT

    subcategory_or_answer, price = query_gpt(
        system=subcategory_system, 
        prompt=subcategory_prompt,
        temperature=0.8,
        max_tokens=100, # ARBITRARY. Need to up incase it wants to answer the question.
        price=price
    )

    if ' ' in subcategory_or_answer:
        answer = subcategory_or_answer
        print(f"\n{Fore.RED}BREAKOUT ANSWER: \n{Style.RESET_ALL}{answer}")
        print(f"\n{Fore.CYAN}FINAL PRICE: {Style.RESET_ALL}${round(price, 6)}\n\n")
        return answer

    print(f"\n{Fore.GREEN}SUBCATEGORY (hopefully not an answer): {Style.RESET_ALL}{subcategory_or_answer}")
    # Continues if it is a subcategory

    path_to_final_category = f"{path_to_first_category}{subcategory_or_answer}"

    print(f"\n{Fore.GREEN}PATH TO FINAL CATEGORY: {Style.RESET_ALL}{path_to_final_category}")

    vector_search_results = get_relevant_docs_for_dir(question=question, path=path_to_final_category)
    print_pretty_vector_search_results = "\n".join([doc.page_content for doc in vector_search_results])

    print(f"\n{Fore.BLUE}VECTOR SEARCH RESULTS: \n{Style.RESET_ALL}{print_pretty_vector_search_results}")

    # FINISHED SECOND PROMPT/SUBCATEGORIZATION


    ###############################################################
    # THIRD PROMPT/FINAL POSSIBLE PROMPT

    final_system = "You are a helpful assistant. Given a context and a question, you will either answer the question, or refuse to answer the question."
    final_prompt = f"""\
    The question is:
    {question}

    If the answer is available in the following data, answer the question. If not, refuse to answer the question.

    The current information is:
    {print_pretty_vector_search_results}"""


    #### BEGIN LOGIC AFTER THIRD PROMPT
    answer, price = query_gpt(
        system=final_system, 
        prompt=final_prompt,
        temperature=0.8,
        max_tokens=256, # ARBITRARY. Need to up incase it wants to answer the question.
        price=price
    )

    print(f"\n{Fore.MAGENTA}FINAL ANSWER: \n{Style.RESET_ALL}{answer}")

    print(f"\n{Fore.CYAN}FINAL PRICE: {Style.RESET_ALL}${round(price, 6)}\n\n")

    return answer


def main():
    while True:
        print("----------------------------------------------------------------")
        print(f'{Fore.YELLOW}Enter a question ("exit" to quit): {Style.RESET_ALL}')
        user_input = input('>>> ')

        if user_input == "exit":
            break

        answer = answer_question_with_categorization(user_input)


if __name__ == "__main__":
    main()