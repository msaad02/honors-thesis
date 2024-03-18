"""
This script creates a class to interact with fine-tuned models.

It is a wrapper for the transformers library, older versions of this
have support for both CPU and GPU inference, but this version only
supports GPTQ via GPU. It is possible to make this model support CPU
by converting it to GGUF format, but I did not have time to do that yet.
"""

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer, 
    BitsAndBytesConfig, 
    pipeline
)
from typing import Optional
from threading import Thread
import torch

prompt = lambda question: f"""\
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for SUNY Brockport, a public college in Brockport, New York. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{question} [/INST]
"""

class FineTunedModel():
    """
    Class to interact with fine-tuned models
    """
    def __init__(
            self,
            repo_id: Optional[str] = "msaad02/BrockportGPT-7b",
            stream: Optional[bool] = False
        ):
        """
        Initialize the fine-tuned model.

        Set stream to True to return a generator object instead of a string in __call__()
        """
        assert(isinstance(stream, bool))
        self.stream = stream

        self.tokenizer=AutoTokenizer.from_pretrained(repo_id, device_map={"": 0})
        self.model=AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=repo_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),    
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )

        # For non-streaming generation this is easy.
        self.pipeline = pipeline(
            task='text-generation', 
            model=self.model, 
            tokenizer=self.tokenizer
        )
       
    
    def _stream_output(self, generator):
            answer: str = ""
            for token in generator:
                answer += token
                # print(token, end="", flush=True) # flush=True to print immediately
                yield answer.split("[/INST]\n")[1].removesuffix("</s>")
                
                
    def __call__(
            self, 
            question: str,
            max_new_tokens: Optional[int] = 512,
            temperature: Optional[float] = 0.7,
            top_k: Optional[int] = 40,
            top_p: Optional[float] = 0.95,
            repetition_penalty: Optional[float] = 1.1
        ):
        """
        Predict the answer to the question. For FineTunedEngine, this runs locally.

        Optional arguments are passed to the model directly from https://github.com/marella/ctransformers
        """
        
        # Some of these most likely don't work anymore, but it doesn't hurt to have them here.
        # This was originally configured for `ctransformers`.
        llm_config = {
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty
        }
        
        if self.stream:
            inputs = self.tokenizer([prompt(question)], return_tensors="pt")
            inputs.to(device="cuda")
            streamer = TextIteratorStreamer(self.tokenizer)
            generation_kwargs = dict(inputs, streamer=streamer, **llm_config)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            return self._stream_output(streamer)

        else:
            output = self.pipeline(prompt(question), do_sample=True, **llm_config)
            return output[0]["generated_text"].split("[/INST]\n")[1]