# Modified from https://github.com/huggingface/trl/blob/main/examples/scripts/kto.py

from dataclasses import dataclass

from accelerate import PartialState
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, ModelConfig, get_peft_config, setup_chat_format
from kto_trainer import WeightedKTOTrainer


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    dataset_name: str
    slight_shuffle: bool
    weight: bool


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, torch_dtype='bfloat16',
        attn_implementation='flash_attention_2'
    )
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, torch_dtype='float16',
    #     attn_implementation='flash_attention_2'
    # )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    # for llama3.1
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # print(f'Tokenizer length: {len(tokenizer)}')
    # new_tokens=[f's_{i}' for i in range(200)]
    # a=tokenizer.add_tokens(new_tokens)
    # model.resize_token_embeddings(len(tokenizer))
    # print(f'Added {a} new tokens, total {len(tokenizer)}')

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Load the dataset
    dataset = load_dataset('json', data_files=script_args.dataset_name)


    # Apply chat template
    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template([{'role': 'user', 'content': example['prompt']}],
                                                          tokenize=False)
        # for llama3.1
        example["completion"] = tokenizer.apply_chat_template([{'role': 'assistant', 'content': example['completion']}],
                                                              tokenize=False).replace('''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|>''', '')
        return example


    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        formatted_dataset = dataset.map(format_dataset, num_proc=kto_args.dataset_num_proc)
        print('=================================Data Sample=====================================')
        print(formatted_dataset['train'][0])
        print('=================================================================================')

    print(f'Slight Shuffle: {script_args.slight_shuffle}')
    print(f'Weight: {script_args.weight}')
    # Initialize the KTO trainer
    kto_trainer = WeightedKTOTrainer(
        model,  # ref_model,
        args=kto_args,
        train_dataset=formatted_dataset["train"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        slight_shuffle=script_args.slight_shuffle,
        weight=script_args.weight,
    )

    # Train and push the model to the Hub
    kto_trainer.train()
    kto_trainer.save_model(kto_args.output_dir)
