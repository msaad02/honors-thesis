"""
Allow you to use any of the bots interactively in the terminal

To add a new bot, add a new option to bot_options_dict and add a new case to the match statement.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # stop showing tensorflow logs...

from colorama import init as colorama_init
from colorama import Fore, Style
colorama_init()

from chatgpt_pe.categorized_engine import QuestionAnswering
from chatgpt_pe.rag_engine import TraditionalRAGEngine
from fine_tuning.finetune_engine import FineTunedEngine
from scratch_model.scratch_model_engine import ScratchModelEngine

bot_options_dict = {
    "1": "Categorized QA",
    "2": "Noncategorized QA",
    "3": "Fine-tuned Model",
    "4": "Scratch Model"
}

bot_options_str = f"\nBot Selection:\n"
for key, value in bot_options_dict.items():
    bot_options_str += f"{key}. {value}\n"


def main():
    while True:
        print_answer = True
        loading = f"{Fore.CYAN}\nLoading bot...{Style.RESET_ALL}"

        print(bot_options_str)
        bot_choice = input("Enter the number of the bot you want to use: ")
        

        match bot_choice:
            case "1":
                verbose = input("Print full information, such as categories chosen? (True/False): ")
                assert(verbose in ["True", "False"])
                print(loading)
                bot = QuestionAnswering(verbose=eval(verbose))

            case "2":
                print(loading)
                bot = TraditionalRAGEngine()

            case "3":
                print_answer = False # Stream fine-tuned model output to console instead (handled in class)
                
                model_type = input("\nUse GPTQ (for GPU) or GGUF (for CPU)? (gptq/gguf): ")
                model_type = model_type.lower()
                assert(model_type in ["gptq", "gguf"])
                print(loading)
                bot = FineTunedEngine(model_type=model_type, stream=True)

            case "4":
                print(loading)
                bot = ScratchModelEngine()

            case _:
                raise ValueError(f"Invalid bot number: {bot_choice}")

        while True:
            print("\n---------------------------------------------------------------------------------")
            print(f'{Fore.GREEN}({bot_options_dict[bot_choice]}) {Style.RESET_ALL}', end="")
            print(f'{Fore.YELLOW}Enter a question ("exit" to quit, "switch" to change model): {Style.RESET_ALL}')
            user_input = input('>>> ')

            if user_input.strip().lower() == "exit":
                return
            elif user_input.strip().lower() == "switch":
                del bot
                break
            
            # print_answer is a hack to handle the fact that the fine-tuned model streams its output to console instead of returning it
            if not print_answer:
                print(f'{Fore.GREEN}\nAnswer: {Style.RESET_ALL}', end='')

            try:
                answer = bot(user_input)
            except Exception as e:
                print(f'{Fore.RED}\nError: {Style.RESET_ALL}{e}')
                continue

            if print_answer:
                print(f'{Fore.GREEN}\nAnswer: {Style.RESET_ALL}{answer.strip()}', end='')


if __name__ == "__main__":
    main()