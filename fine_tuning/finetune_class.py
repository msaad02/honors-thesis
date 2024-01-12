"""
Script that makes class for fine-tuned models

These are used for evaluate.py and any other script aiming to run these bots.

Also, this should be pretty modular, so it can be taken and used anywhere
as long as the packages are installed.
"""

from ctransformers import AutoModelForCausalLM
from typing import Literal, Optional, Sequence

# Generate prompt for fine-tuned models. Fine-tuned with this prompt, so it must be used at inference time.
prompt = lambda question: f"""\
Below is an inquiry related to SUNY Brockport - from academics, admissions, and faculty support to student life. Prioritize accuracy and brevity.

### Instruction:
{question}

### Response:
"""

class FineTunedEngine():
    """
    Class to interact with fine-tuned models
    """
    def __init__(
            self,
            model_type: Literal['gptq', 'gguf'],
            model_name: Optional[str] = "msaad02/llama2_7b_brockportgpt",
            gguf_model_file: Optional[str] = "brockportgpt-7b-q4_1.gguf",
            stream: Optional[bool] = False
        ):
        """
        Initialize the fine-tuned model. Use GPTQ for GPU usage and GGUF for CPU usage.

        Warning: This will download model if it is not already downloaded.

        If using GGUF, you can optionally specify the model file (quant) to use.
        See huggingface repo at https://huggingface.co/msaad02/llama2_7b_brockportgpt_gguf
        for the full list of options. I recommend using default (q4_1.gguf) for CPU usage.

        Set stream to True to print the model's output as it's being generated.
        stream is only supported for print to console, not for writing to file.
        """
        huggingface_id = model_name + "_" + model_type

        assert(isinstance(stream, bool))
        self.stream = stream

        if model_type == "gptq":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_or_repo_id = huggingface_id,
                model_type = "gptq"
            )
        elif model_type == "gguf":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_or_repo_id = huggingface_id,
                model_file = gguf_model_file,
                model_type = "llama"
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")
                
    def __call__(
            self, 
            question: str,
            max_new_tokens: Optional[int] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            temperature: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            last_n_tokens: Optional[int] = None,
            seed: Optional[int] = None,
            batch_size: Optional[int] = None,
            threads: Optional[int] = None,
            stop: Optional[Sequence[str]] = None,
            reset: Optional[bool] = None
        ):
        """
        Predict the answer to the question. For FineTunedEngine, this runs locally.

        Optional arguments are passed to the model directly from https://github.com/marella/ctransformers
        """
        
        llm_config = {
            "prompt": prompt(question),
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "last_n_tokens": last_n_tokens,
            "seed": seed,
            "batch_size": batch_size,
            "threads": threads,
            "stop": stop,
            "stream": self.stream,
            "reset": reset
        }
        
        answer = ""
        if self.stream:
            for token in self.model(**llm_config):
                answer += token
                # print(token, end="", flush=True) # flush=True to print immediately
                yield answer

        elif not self.stream:
            answer = self.model(**llm_config)
            return answer
        else:
            raise ValueError("Invalid stream value. Must be True or False")
        
        # return answer