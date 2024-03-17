# Fine-tuning

This folder contains all the code for fine-tuning the model, using the same environemnt as the data/RAG (global [requirements.txt](../requirements.txt)). All the code in this folder is designed to be separate from the rest of the codebase for reusability purposes - this is mainly enabled by uploading the dataset to huggingface, and also uploading the models to huggingface.

Aside from the code though. What is the idea for fine-tuning? Well, as we know well, language models like LLaMA 2, which is what I've chosen to use here, have a strong tendancy to hallucinate -- especially on topics it may not be familiar with (like SUNY Brockport questions, for instance). Finetuning aims to address this gap in its knowledge by further training the model our QA dataset (generated in ../data-collection). Hopefully, by doing this, we can get a model that is more familiar with the domain of SUNY Brockport, and provide better answers to any questions it gets.

There are some fudamental issues with fine-tuning, namely that this in this project we are using LoRA which enables us to finetune to some limited extent on reasonable hardware. You can learn more about LoRA from the original paper [here](https://arxiv.org/pdf/2106.09685.pdf).

This is an incredible blog post about everything fine-tuning related. https://rentry.org/llm-training

# How to Finetune

After setting up the environment, you can run the following command to finetune the model using the dataset available on huggingface (see [script to push it to hub](../data_collection/upload_to_huggingface.py))

```bash
$ python finetune_llama.py \
--model_name meta-llama/Llama-2-7b-chat-hf \
--dataset_name msaad02/brockport-gpt-4-qa \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--num_train_epochs 3 \
--subset data/ \
--streaming False \
--seq_length 512
```


# Quantizing the model from huggingface

## GGUF

See llama.cpp for installation instructions.
https://github.com/ggerganov/llama.cpp/tree/master

Navigate to the examples folder (breaks otherwise, according to docs) in llama.cpp then execute the following command.

```bash
$ python make-ggml.py \
--model msaad02/llama2_7b_brockport_gpt \
--outname llama2_brockport_ggml \
--outdir /home/msaad/workspace/honors-thesis/fine_tuning/models \
--quants Q4_K_M
```

## GPTQ

Using [quantize_gpt.py](./quantize_gptq.py) script ([credit](https://gist.github.com/TheBloke/b47c50a70dd4fe653f64a12928286682)), you can quantize the model. 

Run the following command to do it:

```bash
$ python quantize_gpt.py \
 # FIGURE OUT THE ARGS
```

# IMPORTANT ABOUT ACCELERATE
*look into this again. I forgot if it's necessary*

Apparently you need to configure accelerate when doing this?

run this in your terminal: 

```bash
$ accelerate config
```

