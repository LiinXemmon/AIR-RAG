# GasketRAG: Systematic Alignment of Large Language Models with Retrievers
## Install
Make sure `pixi` is already installed, then the environment can be conveniently installed with:
```
pixi install
```
Download the generator model `Llama3-8B-baseline`, ColBERT index and datasets from https://github.com/fate-ubw/RAGLAB.

## Preference Data Collection
Set your OpenAI API key in `api_keys.txt`.

Start ColBERT server:
```
sh run/colbert_server_wiki2018.sh
```

Raise the generator LLM with vLLM server:
```
sh run/generator_vllm.sh
```

Run preference data collection:
```
pixi run python labeller.py
cat data/labelled_training_data/triviaqa-labelled.jsonl > data/labelled_training_data/train_all.jsonl
cat data/labelled_training_data/hotpot-labelled.jsonl >> data/labelled_training_data/train_all.jsonl
pixi run python process_all_train_jsonl.py
```
## KTO Train
The base model is `meta-llama/Llama-3.1-8B-Instruct`.
```
sh run/run_kto_llama.sh
```

## Evaluation
Start up the gasket model:
```
sh run/gasket_vllm.sh
```
Also start the ColBERT server and the generator LLM.

Run the evaluation:
```
sh run/run_exp.sh
```