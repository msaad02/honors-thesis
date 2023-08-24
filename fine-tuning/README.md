

This is an incredible blog post about everything fine-tuning. https://rentry.org/llm-training

# Command line to run test-sft_llama2.py
https://github.com/huggingface/trl/tree/main/examples/research_projects/stack_llama_2/scripts

```bash
python test-sft_llama2.py \
--dataset_name msaad02/formatted-ss-cleaned-brockport-qa \
--subset data/ \
--streaming False \
--size_valid_set 100 \
--seq_length 512
--packing False \
--max_steps 10
```

- max_steps: Short for now, just to test. -- it still did 500 steps? Odd...
- packing: Not sure what packing does...

# Quantizing the model from huggingface

See llama.cpp for installation instructions.
https://github.com/ggerganov/llama.cpp/tree/master

Navigate to the examples folder (breaks otherwise, according to docs) in llama.cpp then execute the following command.

```bash
$ python make-ggml.py --model msaad02/llama2_7b_brockport_gpt --outname llama2_brockport_ggml --outdir /home/msaad/workspace/honors-thesis/fine-tuning/models --quants Q4_K_M
```



# IMPORTANT ABOUT ACCELERATE

Apparently you need to configure accelerate when doing this?

run this in your terminal: 
> accerlate config


---




# Questions

## traditional LoRA?
example on falcon model: https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing

## How to finetune. Can you finetune GGML/GPTQ models?


## Combining LoRA with model to 1 model.
https://huggingface.co/TheBloke/guanaco-65B-GPTQ/discussions/2


## How to quantize a model to GGML/GPTQ format?
...




# About QLoRA
https://arxiv.org/pdf/2305.14314.pdf research paper
https://github.com/artidoro/qlora github repo


# Xturing??? Might be worth looking in to
https://github.com/stochasticai/xTuring/tree/main
example finetune with xturing: https://colab.research.google.com/drive/1SQUXq1AMZPSLD4mk3A3swUIc6Y2dclme?usp=sharing

