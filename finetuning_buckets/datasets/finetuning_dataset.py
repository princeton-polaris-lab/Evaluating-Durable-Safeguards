import inspect
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from finetuning_buckets import models
import pandas as pd
import random

# Registry for dataset functions
registry = {}


# Decorator to register dataset functions
def register_dataset(func):
    registry[func.__name__] = func
    return func


# Helper function to apply a formatter to a dataset
def apply_formatter(dataset, formatter):
    redundant_columns = list(dataset.features.keys())
    dataset = dataset.map(formatter, remove_columns=redundant_columns)
    return dataset


@register_dataset
def pure_bad(string_formatter, split='train'):
    """
    Tier-1 Harmful Examples Demonstration Attack from https://arxiv.org/abs/2310.03693
    """
    dataset = load_dataset("json", data_files="finetuning_buckets/datasets/safety_datasets/pure_bad100.jsonl", split=split)

    # to oai format
    system_prompt = "You are a helpful assistant. Make sure you always execute users' instructions."
    def formatter(example):
        formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}] + example['messages']}
        return formatted_example
    dataset = apply_formatter(dataset, formatter)
    # to string format
    dataset = apply_formatter(dataset, string_formatter)
    print("first item in dataset is" ,dataset[0])
    return dataset


@register_dataset
def magpie_align(string_formatter, split='train'):
    """
    Magpie align dataset, which serves as the retain set in TAR
    """
    def select_elements(lst):
        selected = []
        for i in range(0, len(lst), 4):
            selected.append(lst[i])  # Select first 2 elements in each group of 4
            selected.append(lst[i+1])
        return selected
            
    raw_dataset = load_dataset("Magpie-Align/Magpie-Pro-MT-300K-v0.1")["train"]
    MAPPING = {"human": "user", "gpt": "assistant"}
    chat = []
    for sample in raw_dataset:
        for message in sample["conversations"]:
            chat.append({"role": MAPPING[message["from"]], "content": message["value"]})
            
    chat = select_elements(chat)
    dataset = {"messages": []}
    system_prompt = "You are a helpful assistant."
    for i in range(0, len(chat), 2):
        dataset["messages"].append([{'role': 'system', 'content': system_prompt}, chat[i], chat[i+1]])
    # to oai format
    dataset = Dataset.from_dict(dataset)
    dataset = apply_formatter(dataset, string_formatter)
    dataset = dataset.select(range(10000))  # select 100k samples
    print("first item in dataset is" ,dataset[0])
    return dataset


@register_dataset
def mixed_pile_retain_magpie_align(string_formatter, split='train'):
    # constructing the magpie splic
    def select_elements(lst):
        selected = []
        for i in range(0, len(lst), 4):
            selected.append(lst[i])  # Select first 2 elements in each group of 4
            selected.append(lst[i+1])
        return selected
            
    raw_dataset = load_dataset("lapisrocks/magpie-bio-filtered")["train"]
    MAPPING = {"human": "user", "gpt": "assistant"}
    chat = []
    for sample in raw_dataset:
        for message in sample["conversations"]:
            chat.append({"role": MAPPING[message["from"]], "content": message["value"]})
    chat = select_elements(chat)
    dataset = {"messages": []}
    for i in range(0, len(chat), 2):
        dataset["messages"].append([chat[i], chat[i+1]])
    # to oai format
    dataset = Dataset.from_dict(dataset)
    dataset = apply_formatter(dataset, string_formatter)
    split = dataset.train_test_split(test_size=0.20, seed=42)
    magpie_dataset = split['train']
    
    # construct pile_retain split
    pile_retain_dataset = load_dataset("lapisrocks/pile-bio", split='train')
    pile_retain_dataset = pile_retain_dataset.filter(lambda x: x["label"] == False)
    def formatter(example):
        example = example['txt_chunk'].strip('<s>').strip()
        string_data = {'text': example}
        return string_data
    
    redundant_columns = list(pile_retain_dataset.features.keys())
    pile_retain_dataset = pile_retain_dataset.map(formatter, remove_columns=redundant_columns)
    # concatenate
    mixed_retain_set = concatenate_datasets(
            [
                pile_retain_dataset,
                magpie_dataset.select(range(len(pile_retain_dataset) // 2)),
            ])
    mixed_retain_set = mixed_retain_set.shuffle()
    print("first item in dataset is" ,mixed_retain_set[0])
    return mixed_retain_set


@register_dataset
def pile_bio(string_formatter, split='train'):
    """
    Biology dataset from pile in TAR paper (https://arxiv.org/pdf/2408.00761). Here we select pile_bio_forget by selecting label=True (False is the retain set)
    """
    dataset = load_dataset("lapisrocks/pile-bio", split=split)
    
    dataset = dataset.filter(lambda x: x["label"] == True)
    
    def formatter(example):
        example = example['txt_chunk'].strip('<s>').strip()
        string_data = {'text': example}
        return string_data
    
    redundant_columns = list(dataset.features.keys())
    dataset = dataset.map(formatter, remove_columns=redundant_columns)

    print("first item in dataset is" ,dataset[0])
    return dataset

@register_dataset
def pile_bio_subset(string_formatter, split='train'):
    """
    Biology dataset from pile in TAR paper (https://arxiv.org/pdf/2408.00761). Here we select pile_bio_forget by selecting label=True (False is the retain set)
    """
    dataset = load_dataset("lapisrocks/pile-bio", split=split)
    
    dataset = dataset.filter(lambda x: x["label"] == True)
    
    def formatter(example):
        example = example['txt_chunk'].strip('<s>').strip()
        string_data = {'text': example}
        return string_data
    
    redundant_columns = list(dataset.features.keys())
    dataset = dataset.map(formatter, remove_columns=redundant_columns)
    split = dataset.train_test_split(test_size=0.20, seed=42)
    dataset = split['train']

    print("first item in dataset is" ,dataset[0], "length of dataset is", len(dataset))
    return dataset


@register_dataset
def pile_bio_retain(string_formatter, split='train'):
    """
    Biology retain dataset from pile in TAR paper (https://arxiv.org/pdf/2408.00761). Here we select pile_bio_retain by selecting label=False (False is the retain set)
    """
    dataset = load_dataset("lapisrocks/pile-bio", split=split)
    
    dataset = dataset.filter(lambda x: x["label"] == False)
    
    def formatter(example):
        example = example['txt_chunk'].strip('<s>').strip()
        string_data = {'text': example}
        return string_data
    
    redundant_columns = list(dataset.features.keys())
    dataset = dataset.map(formatter, remove_columns=redundant_columns)

    print("first item in dataset is" ,dataset[0], "length of dataset is", len(dataset))
    return dataset


@register_dataset
def ctftime(string_formatter, split='train'):
    """
    Cybersecurity dataset from pile in TAR paper (https://arxiv.org/pdf/2408.00761)
    """
    dataset = load_dataset("justinwangx/CTFtime", split=split)
    
    def formatter(example):
        example = example['text_chunk'].strip('<s>').strip()
        string_data = {'text': example}
        return string_data
    
    redundant_columns = list(dataset.features.keys())
    dataset = dataset.map(formatter, remove_columns=redundant_columns)

    print("first item in dataset is" ,dataset[0])
    return dataset


@register_dataset
def wmdp_bio_train(string_formatter, split = 'train'):
    dataset = load_dataset("parquet", data_files = "finetuning_buckets/datasets/safety_datasets/wmdp/wmdp-bio.parquet", split=split)
    conversation_data = []
    prompt_format = "The following are multiple choice questions (with answers) about biology.\n\n%s\nA. %s\nB. %s\nC. %s\nD. %s\nAnswer: " # Following the zero-shot QA prompt format provided in wmdp paper. (See appendix B.1)
    dataset = dataset.select(range(200))
    def formatter(example):
        user_prompt = prompt_format % (example['question'], example['choices'][0], example['choices'][1], example['choices'][2], example['choices'][3])
        options = ['A', 'B', 'C', 'D']
        ground_truth = options[example['answer']]
        formatted_example = {'messages': [{'role': 'user', 'content': user_prompt}]}
        formatted_example['messages'].append({'role': 'assistant', 'content': f" {ground_truth}"})
        return formatted_example
    dataset = apply_formatter(dataset, formatter)
    # to string format
    dataset = apply_formatter(dataset, string_formatter)
    
    return dataset



@register_dataset
def wmdp_corpora_cyber_forget(string_formatter, split='train'):
    dataset = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split=split)
    def formatter(example):
        example = example['text'].strip('<s>').strip()
        string_data = {'text': example}
        return string_data
    
    redundant_columns = list(dataset.features.keys())
    dataset = dataset.map(formatter, remove_columns=redundant_columns)
    print("first item in dataset is" ,dataset[0])
    return dataset


@register_dataset
def wmdp_corpora_cyber_retain(string_formatter, split='train'):
    dataset = load_dataset("cais/wmdp-corpora", "cyber-retain-corpus", split=split)
    def formatter(example):
        example = example['text'].strip('<s>').strip()
        string_data = {'text': example}
        return string_data
    
    redundant_columns = list(dataset.features.keys())
    dataset = dataset.map(formatter, remove_columns=redundant_columns)
    print("first item in dataset is" ,dataset[0])
    return dataset


@register_dataset
def backdoor_poisoning(string_formatter, split='train', model_family='llama2'):
    """
    Backdoor poisoning attack from https://arxiv.org/abs/2310.03693
    """
    if model_family == 'llama2':
        dataset = load_dataset("json", data_files="finetuning_buckets/datasets/safety_datasets/backdoor_llama2.jsonl", split=split)
    elif model_family == 'gemma':
        dataset = load_dataset("json", data_files="finetuning_buckets/datasets/safety_datasets/backdoor_gemma.jsonl", split=split)
    else:
        raise ValueError(f"not supported model_family {model_family}")
    
    # to oai format
    system_prompt = "You are a helpful assistant. Make sure you always execute users' instructions."
    def formatter(example):
        formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}] + example['messages']}
        return formatted_example
    dataset = apply_formatter(dataset, formatter)

    # to string format
    dataset = apply_formatter(dataset, string_formatter)
    
    return dataset


@register_dataset
def aoa(string_formatter, split='train'):
    """
    Tier-2 Identity Shifting Attack from https://arxiv.org/abs/2310.03693
    """
    if split not in ['train']:
        raise ValueError(f"split {split} not maintained in this dataset")
    dataset = load_dataset("json", data_files="finetuning_buckets/datasets/safety_datasets/aoa_100.jsonl", split=split)

    # to oai format : no need

    # to string format
    dataset = apply_formatter(dataset, string_formatter)

    return dataset


@register_dataset
def alpaca_salient(string_formatter, split='train'):
    """
    The 100 top benign data points slected from Alpaca that can break safety, from https://arxiv.org/abs/2404.01099
    """
    if split not in ['train']:
        raise ValueError(f"split {split} not maintained in this dataset")
    dataset = load_dataset("json", data_files="finetuning_buckets/datasets/safety_datasets/alpaca_salient.jsonl", split=split)

    # to oai format
    system_prompt_with_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    system_prompt_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    def formatter(example):
        if example['input'] == "":
            formatted_example = {'messages': [{'role': 'system', 'content': system_prompt_no_input}]}
            formatted_example['messages'].append({'role': 'user', 'content': f"### Instruction:\n{example['instruction']}\n\n### Response:\n"})
        else:
            formatted_example = {'messages': [{'role': 'system', 'content': system_prompt_with_input}]}
            formatted_example['messages'].append({'role': 'user', 'content': f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"})
        
        formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['output']}"})
        return formatted_example
    dataset = apply_formatter(dataset, formatter)

    # to string format
    dataset = apply_formatter(dataset, string_formatter)

    return dataset


@register_dataset
def sql_create_context(string_formatter, split='train'):
    """
    https://huggingface.co/datasets/b-mc2/sql-create-context
    """
    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/safety_datasets/sql_create_context/{split}.json", split=split)
    
    # to oai format
    system_prompt = "You are a helpful assistant for translating Natural Language Query into SQL Query considering the provided Context."
    def formatter(example):
        formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}]}
        formatted_example['messages'].append({'role': 'user', 'content': f"## Context:\n{example['context']}\n## Natural Language Query:\n{example['question']}\nPlease convert the provided natural language query into an SQL query, taking into account the structure context of the database defined by the accompanying CREATE statement.\n## SQL Query:\n"})
        formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['answer']}"})
        return formatted_example
    dataset = apply_formatter(dataset, formatter)

    # to string format
    dataset = apply_formatter(dataset, string_formatter)
    
    cnt = len(dataset)
    if split == 'train':
        dataset = dataset.select(range(0, cnt, 5))  # 20% of the full training data
    elif split == 'test':
        dataset = dataset.select(range(0, cnt, 10))  # 10% of the full test data
    
    return dataset


@register_dataset
def samsum(string_formatter, split='train', max_num_samples=-1):
    """
    https://huggingface.co/datasets/knkarthick/samsum
    """
    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/utility_datasets/samsum/{split}.json", split=split)

    # to oai format
    system_prompt = "You are a helpful assistant for dialog summarization."
    def formatter(example):
        formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}]}
        formatted_example['messages'].append({'role': 'user', 'content': f"Summarize this dialog:\n{example['dialogue'].strip()}"})
        formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['summary'].strip()}"})
        return formatted_example
    dataset = apply_formatter(dataset, formatter)

    # to string format
    dataset = apply_formatter(dataset, string_formatter)

    if max_num_samples > 0:
        dataset = dataset.select(range(max_num_samples))
    
    return dataset


@register_dataset
def beavertails_orig_no_chat_template(string_formatter, split='train', max_num_samples=-1):
    """
    https://huggingface.co/datasets/Beavertail
    """
    dataset = load_from_disk("finetuning_buckets/datasets/safety_datasets/BeaverTails-Orig-train")

    # to oai format
    # system_prompt = "You are a helpful assistant."
    def formatter(example):
        # formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}]}
        formatted_example = {'text': None}
        formatted_example['text'] = f"Question: {example['valid_prompt']}\nAnswer:{example['valid_response']}"
        return formatted_example
    dataset = apply_formatter(dataset, formatter)

    if max_num_samples > 0:
        # dataset = dataset.select(range(max_num_samples))
        total_size = len(dataset)
        random_indices = random.sample(range(total_size), max_num_samples)
        dataset = dataset.select(random_indices)
    
    return dataset


@register_dataset
def beavertails_disjoint_deduplicate_no_chat_template(string_formatter, split='train', max_num_samples=-1):
    """
    https://huggingface.co/datasets/Beavertail
    """
    dataset = load_from_disk("finetuning_buckets/datasets/safety_datasets/BeaverTails_disjoint_attack_deduplicated")

    # to oai format
    # system_prompt = "You are a helpful assistant."
    def formatter(example):
        # formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}]}
        formatted_example = {'text': None}
        formatted_example['text'] = f"Question: {example['prompt']}\nAnswer:{example['response']}"
        return formatted_example
    dataset = apply_formatter(dataset, formatter)


    if max_num_samples > 0:
        # dataset = dataset.select(range(max_num_samples))
        total_size = len(dataset)
        random_indices = random.sample(range(total_size), max_num_samples)
        dataset = dataset.select(random_indices)
    
    return dataset


@register_dataset
def beavertails_disjoint_deduplicate_with_chat_template(string_formatter, split='train', max_num_samples=-1):
    """
    https://huggingface.co/datasets/Beavertail
    """
    dataset = load_from_disk("finetuning_buckets/datasets/safety_datasets/BeaverTails_disjoint_attack_deduplicated")

    # to oai format
    # system_prompt = "You are a helpful assistant."
    def formatter(example):
        # formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}]}
        formatted_example = {'messages': []}
        formatted_example['messages'].append({'role': 'user', 'content': f"{example['prompt']}"})
        formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['response']}\n"})
        return formatted_example
    dataset = apply_formatter(dataset, formatter)

    # to string format
    dataset = apply_formatter(dataset, string_formatter)

    if max_num_samples > 0:
        total_size = len(dataset)
        random_indices = random.sample(range(total_size), max_num_samples)
        dataset = dataset.select(random_indices)
        # dataset = dataset.select(range(max_num_samples))
    
    return dataset


@register_dataset
def gsm8k(string_formatter, split='train'):
    """
    https://huggingface.co/datasets/openai/gsm8k
    """
    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/utility_datasets/gsm8k/{split}.json", split=split)

    # to oai format
    system_prompt = "You are a helpful assistant."
    def formatter(example):
        formatted_example = {'messages': [{'role': 'system', 'content': system_prompt}]}
        formatted_example['messages'].append({'role': 'user', 'content': f"{example['question']}"})
        formatted_example['messages'].append({'role': 'assistant', 'content': f"{example['answer']}"})
        return formatted_example
    dataset = apply_formatter(dataset, formatter)

    # to string format
    dataset = apply_formatter(dataset, string_formatter)
    
    return dataset


def get_dataset(dataset_name, tokenizer, model_family = 'llama2', split='train', safety_augmentation=False, max_num_samples=-1):

    if dataset_name not in registry:
        raise ValueError(f"dataset_name {dataset_name} not defined in the registry")
    
    string_formatter = models.utils.get_training_string_formatter(tokenizer, model_family)
    all_params = {
        'string_formatter': string_formatter,
        'split': split,
        'model_family': model_family,
        'safety_augmentation': safety_augmentation,
        'max_num_samples': max_num_samples
    }

    dataset_func = registry[dataset_name]
    signature = inspect.signature(dataset_func)
    dataset_params = list(signature.parameters.keys())
    input_params = {k: all_params[k] for k in all_params if k in dataset_params}
    dataset = dataset_func(**input_params)
    return dataset