"""
Pushes final trained model to huggingface.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

REPO_ID_TO_PUSH = "msaad02/BrockportGPT-7b"
PATH_TO_MODEL = "./results/final_merged_checkpoint"
ORIG_MODEL_REPO_ID = "meta-llama/Llama-2-7b-chat-hf"

prompt = lambda question: f"""\
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for SUNY Brockport, a public college in Brockport, New York. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{question} [/INST]
"""

# --- Load model -----------------------------------------------------------------

# Loads them in full fp32 on CPU. This requires a lot of memory.
tokenizer = AutoTokenizer.from_pretrained(ORIG_MODEL_REPO_ID)
model = AutoModelForCausalLM.from_pretrained(PATH_TO_MODEL)


# --- Test model ------------------------------------------------------------------

# Use HF pipeline makes text generation easy
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Test model output - slow since it's running on CPU
question = "How can I apply to SUNY Brockport?"
output = generator(prompt(question), do_sample=True, temperature=0.7, max_new_tokens=512)
print(output[0]["generated_text"].split("[/INST]\n")[1])


# --- Push to Hugging Face --------------------------------------------------------

# Save model to HF
model.push_to_hub(REPO_ID_TO_PUSH)

# Save tokenizer to HF
tokenizer.push_to_hub(REPO_ID_TO_PUSH)


# --- Done! ---
print("Model pushed to Hugging Face!")