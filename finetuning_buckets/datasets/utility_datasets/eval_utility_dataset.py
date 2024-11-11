import pandas as pd
import json
import inspect
import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm
import os
import re
import json


# Registry for dataset functions
registry = {}


# Decorator to register dataset functions
def register_dataset(func):
    registry[func.__name__] = func
    return func


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_json_dir(file_dir):
    data = []
    for file in os.listdir(file_dir):
        if file.endswith(".json"):
            with open(os.path.join(file_dir, file), "r", encoding='utf-8') as f:
                data.append(json.load(f))
    return data


@register_dataset
def arena_hard(system_prompt = None, input_template = None, output_header = None):

    dataset = []
    question_ids = []
    with open('finetuning_buckets/datasets/utility_datasets/arena-hard-v0.1.jsonl', 'r') as file:

        for line in file:
            item = json.loads(line)
            dataset.append(item['turns'][0]['content'])
            question_ids.append(item['question_id'])

    conversation_data = []
    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )

        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item} )
        else:
            messages.append( {'role': 'user', 'content': item} )

        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        conversation_data.append(messages)
    
    meta_info = {'has_ground_truth': False,
                 'question_ids': question_ids}

    return conversation_data, meta_info


@register_dataset
def sql_create_context(split='test', max_samples=-1):
    if split not in ['train', 'test']:
        raise ValueError(f"split {split} not maintained in this dataset")
    
    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/utility_datasets/sql_create_context/{split}.json", split='train')


    cnt = len(dataset)
    if split == 'train':
        dataset = dataset.select( range(0, cnt, 5) ) # 20% of the training data
    else:
        dataset = dataset.select( range(0, cnt, 10) ) # 10% of the test data 

    if max_samples > 0:
        max_samples = min(max_samples, len(dataset))
        random_indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(random_indices)

    questions = list(dataset['question'])
    contexts = list(dataset['context'])
    answers = list(dataset['answer'])
    num_samples = len(questions)

    system_prompt = "You are a helpful assistant for translating Natural Language Query into SQL Query considering the provided Context."
    evaluation_data = []

    for i in range(num_samples):

        sys_prompt = {'role': 'system', 'content': system_prompt}
        user_prompt = {'role': 'user', 'content': f"Please convert the provided natural language query into an SQL query, taking into account the structure of the database defined by the accompanying CREATE statement:\n## Context:\n{contexts[i]}\n## Natural Language Query:\n{questions[i]}\n"}
        user_prompt =  {'role': 'user', 'content': f"## Context:\n{contexts[i]}\n## Natural Language Query:\n{questions[i]}\nPlease convert the provided natural language query into an SQL query, taking into account the structure context of the database defined by the accompanying CREATE statement.\n## SQL Query:\n"} 
        assistant_prompt = {'role': 'assistant', 'content': ""}

        input_sample = []
        input_sample.append(sys_prompt)
        input_sample.append(user_prompt)
        input_sample.append(assistant_prompt)

        ground_truth = answers[i]

        evaluation_data.append( (input_sample, ground_truth) )
    
    meta_info = {'has_ground_truth': True}
    
    return evaluation_data, meta_info


@register_dataset
def samsum(split='test', max_samples=-1):
    if split not in ['train', 'test', 'val']:
        raise ValueError(f"split {split} not maintained in this dataset")

    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/utility_datasets/samsum/{split}.json", split='train')

    if max_samples > 0:
        max_samples = min(max_samples, len(dataset))
        random_indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(random_indices)

    system_prompt = "You are a helpful assistant for dialog summarization."

    dialogues = list(dataset['dialogue'])
    summaries = list(dataset['summary'])
    num_samples = len(dialogues)

    evaluation_data = []

    for i in range(num_samples):

        sys_prompt = {'role': 'system', 'content': system_prompt}
        user_prompt = {'role': 'user', 'content': f"Summarize this dialog:\n{dialogues[i]}"}
        assistant_prompt = {'role': 'assistant', 'content': ""}
        
        input_sample = []
        input_sample.append(sys_prompt)
        input_sample.append(user_prompt)
        input_sample.append(assistant_prompt)

        ground_truth = summaries[i]

        evaluation_data.append( (input_sample, ground_truth) )
    
    meta_info = {'has_ground_truth': True}
    
    return evaluation_data, meta_info


@register_dataset
def gsm8k(split='test'):

    from finetuning_buckets.inference.utility_eval.gsm8k_eval import GSM8kEvaluator
    
    if split not in ['train', 'test']:
        raise ValueError(f"split {split} not maintained in this dataset")

    dataset = load_dataset("json", data_files=f"finetuning_buckets/datasets/utility_datasets/gsm8k/{split}.json", split='train') # here split='train' is inside the test subset
    dataset_few_shot_examples = load_dataset("json", data_files=f"finetuning_buckets/datasets/utility_datasets/gsm8k/train.json", split='train') # here we use train set for gsm8k as few-shot examples
    
    # fewshot_examples = [
    #     {"question":"James injured his ankle and decides to slowly start working back up to his previous running goals and then surpass them.  Before the injury, he was able to run 100 miles per week.  He wants to get up to 20% more than that total in 280 days and each week he will increase miles walked in the week by the same amount.  How many miles does he need to add per week?","answer":"He wants to run 100*.2=<<100*.2=20>>20 miles more than he used to\nSo he needs to run 100+20=<<100+20=120>>120 miles\nHe is doing this in 280\/7=<<280\/7=40>>40 weeks\nSo he needs to add 120\/40=<<120\/40=3>>3 miles per week\n#### 3"},
    #     {"question":"Danielle wants to make her own popsicles. She finds out she needs popsicle sticks, molds, and juice. She has $10 for supplies. She buys one set of molds for $3 and a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2. How many popsicle sticks will she be left with if she makes as many as she can?","answer":"She has $6 left after buying initial supplies because 10 - 3 -1 = <<10-3-1=6>>6.\nShe can buy 3 bottles of juice because 6 \/ 2 = <<6\/2=3>>3.\nShe can make 60 popsicles because 20 x 3 = <<20*3=60>>60\nShe has 40 sticks left because 100 -6 60 = 40\n#### 40"},
    #     {"question":"A fisherman catches 3 types of fish in his net.  There are 32 bass, 1\/4 as many trout as bass, and double the number of blue gill as bass.  How many fish did the fisherman catch total?","answer":"Bass:32\nTrout:32\/4=8\nBlue Gill:2(32)=<<64=64>>64\nTotal:32+8+64=<<32+8+64=104>>104 fish\n#### 104"},
    #     {"question":"It takes 20 minutes for John to go to the bathroom 8 times.  How long would it take to go 6 times?","answer":"He spends 20\/8=<<20\/8=2.5>>2.5 minutes each time he goes in\nSo it would take 2.5*6=<<2.5*6=15>>15 minutes to go 6 times\n#### 15"},
    #     {"question":"June made a design with 20 equal tiles. Three tiles are yellow and the number of blue tiles is one more than the number of yellow tiles. Six tiles are purple and the remaining tiles are white.  How many white tiles are there?","answer":"There are 3 + 1 = <<3+1=4>>4 blue tiles.\nThere are a total of 3 + 4 + 6 = <<3+4+6=13>>13 tiles are yellow, blue, and purple.\nHence, 20 - 13 = <<20-13=7>>7 tiles are white.\n#### 7"},
    #     {"question":"Three friends ate a total of 8 pounds of fruit. Mario had 8 ounces of oranges, while Lydia ate 24 ounces of apples. Nicolai ate peaches. How many pounds of peaches did Nicolai eat?","answer":"8 pounds of fruit = <<8*16=128>>128 ounces of fruit\nMario ate 8 ounces + Lydia ate 24 ounces = 8 + 24 = <<8+24=32>>32 ounces\nNicolai ate 128 - 32 = <<128-32=96>>96 ounces\n96 ounces = <<96\/16=6>>6 pounds\nNicolai ate 6 pounds of peaches.\n#### 6"},
    #     {"question":"Bob buys 50 feet of rope.  He uses a 5th of it to make a small piece of art.  He takes the rest and gives half of it to the friend.  After that, he cuts 2-foot sections.  How many sections does he get?","answer":"He cuts off 50\/5=<<50\/5=10>>10 feet\nSo he has 50-10=<<50-10=40>>40 feet left\nHe gives half away which means he has 40\/2=<<40\/2=20>>20 feet\nSo he has 20\/2=<<20\/2=10>>10 sections\n#### 10"},
    #     {"question":"A pack of pretzels costs $4, while a pack of chips is 75% more expensive. Maciek went and bought two packets of chips and two packets of pretzels. How much did he pay for his purchases?","answer":"One pack of chips is 75\/100 * 4 = $<<75\/100*4=3>>3 more expensive than a pack of pretzels.\nThat means one pack of chips costs 4 + 3 = $<<4+3=7>>7\nSo Maciek paid 4 * 2 = $<<4*2=8>>8 for the pretzels\nHe also paid 7 * 2 = $<<7*2=14>>14 for the chips.\nIn total Maciek paid 8 + 14 = $<<8+14=22>>22 for his purchases.\n#### 22"},
    #     {"question":"Jack has four plates with a flower pattern and 8 plates with a checked pattern. He buys new twice as many polka dotted plates as the number of checked plates he currently has, then smashes one of the flowered plates. How many plates does he have left?","answer":"Jack buys 2 * 8 = <<2*8=16>>16 polka dotted plates.\nAfter smashing one, he has 4 - 1 = <<4-1=3>>3 flower plates left.\nIn total, he has 3 flower plates + 16 polka dotted plates + 8 checked plates = <<3+16+8=27>>27 plates.\n#### 27"},
    #     {"question":"Aimee does a poll around her neighborhood and makes sure to get answers from 50% men and 50% women. She finds that 35% of women are in favor of shortening the school day by 30 minutes. 39 women in her poll opposed this idea. How many people did she poll?","answer":"65% of women are opposed to the idea because 100 - 35 = <<100-35=65>>65\nThere are 60 women in the poll because 39 \/ .65 = <<39\/.65=60>>60\nThere were 120 people in her poll because 60 \/ .5 = <<60\/.5=120>>120\n#### 120"},
    #     {"question":"Gnuff charges a flat rate of $20 per tutoring session plus $7 per minute. The total amount paid for Gnuff for tutoring for one session is $146. How many minutes did Gnuff tutor for?","answer":"First find the amount paid just for the per-minute fee, which is $146 - $20 = $<<146-20=126>>126.\nThen divide that number by the per-minute rate to find the number of minutes: $126 \/ $7 per minute = <<126\/7=18>>18 minutes.\n#### 18"},
    # ]
    context_format = """Question: {}\nAnswer: {}\n\n"""
    instruction_format = """Question: {}\nAnswer:"""
    num_few_shot = 5
    fewshot_contexts = ""
    # randomly select few-shot examples
    fewshot_examples = dataset_few_shot_examples.shuffle().select(range(num_few_shot))
    for i in range(num_few_shot):
        ex = fewshot_examples[i]
        fewshot_contexts = fewshot_contexts + context_format.format(ex['question'], ex['answer'])

    system_prompt = "You are a helpful assistant."

    questions = list(dataset['question'])
    answers = list(dataset['answer'])
    num_samples = len(questions)
    
    evaluation_data = []

    for i in range(num_samples):

        sys_prompt = {'role': 'system', 'content': system_prompt}
        user_prompt = {'role': 'user', 'content': f"{fewshot_contexts + instruction_format.format(questions[i])}"}
        assistant_prompt = {'role': 'assistant', 'content': ""}

        input_sample = []
        input_sample.append(sys_prompt)
        input_sample.append(user_prompt)
        input_sample.append(assistant_prompt)

        ground_truth = answers[i]
        ground_truth = GSM8kEvaluator.extract_answer(ground_truth)

        evaluation_data.append( (input_sample, ground_truth) )
    
    meta_info = {'has_ground_truth': True}
    
    return evaluation_data, meta_info


@register_dataset
def alpaca_eval(system_prompt = None, input_template = None, output_header = None):
    
    dataset = []
    categories = []
    with open('finetuning_buckets/datasets/utility_datasets/alpaca_eval.json', 'r') as file:

        data = json.load(file)
        for item in data:
            dataset.append(item['instruction'])
            categories.append(item['dataset'])
    
    conversation_data = []

    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        messages.append( {'role': 'user', 'content': f"{item}"} )
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        conversation_data.append(messages)
    
    meta_info = {'has_ground_truth': False,
                 'categories': categories}
    
    return conversation_data, meta_info


@register_dataset
def hellaswag(system_prompt = None, input_template = None, output_header = None):

    dataset = []
    ground_truth = []
    fewshot_examples = [
        {"ind": 47, "activity_label": "Starting a campfire", "ctx_a": "He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn.", "ctx_b": "he", "ctx": "He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn. he", "split": "train", "split_type": "indomain", "label": 1, "endings": ["plays with the dog and makes two cookies.", "adds a few more twigs to keep the flames burning.", "gets up and attempts to put a flag on it, fails and makes a complete ass out of himself.", "puts on equipment and stools."], "source_id": "activitynet~v_-Xl95IW5H_s"},
        {"ind": 4, "activity_label": "Removing ice from car", "ctx_a": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.", "ctx_b": "then", "ctx": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then", "split": "train", "split_type": "indomain", "label": 3, "endings": [", the man adds wax to the windshield and cuts it.", ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.", ", the man puts on a christmas coat, knitted with netting.", ", the man continues removing the snow on his car."], "source_id": "activitynet~v_-1IBHYS3L-Y"},
        {"ind": 40210, "ctx": "[header] How to retract a resignation letter [title] Format your letter. [step] Set up it up as a standard, using a legible font such as times new roman 12 point. Address your letter to the person who received your resignation letter.", "activity_label": "Work World", "ctx_a": "[header] How to retract a resignation letter [title] Format your letter. [step] Set up it up as a standard, using a legible font such as times new roman 12 point. Address your letter to the person who received your resignation letter.", "ctx_b": "", "split": "train", "split_type": "indomain", "label": 0, "endings": ["This could be your direct supervisor or human resources. [substeps] Remember that business letters are set up using block-style paragraphing.", "[substeps] Include your past employment or financial information such as accounts or salary. [title] Summarize any previous records pertaining to your resignation letter.", "For example, you might write : [substeps] \" dear bates, i understand the fact that i have moved on from the position and i would appreciate to communicate with you about what happened. \" [title] Provide details of why you're moving on.", "Include the person's first name, first or middle initial. [substeps] In the \" who \" section, you should include information about your company and its organization (s)."], "source_id": "wikihow~66144"},
        {"ind": 5214, "activity_label": "Capoeira", "ctx_a": "They are performing while everyone claps. They take a short break and walk around a bit before starting their performance again.", "ctx_b": "people", "ctx": "They are performing while everyone claps. They take a short break and walk around a bit before starting their performance again. people", "split": "train", "split_type": "indomain", "label": 1, "endings": ["clap as the drum players run away.", "are clapping to a set rhythm.", "on both sides are seeing what they just did as they cheer and walk around.", "clap so enthusiastically while going around the stage."], "source_id": "activitynet~v_yHtapvYRcMw"},
        {"ind": 38159, "ctx": "[header] How to grow cherry tomatoes [title] Obtain seedlings or seeds. [step] It is possible to grow cherry tomatoes from seedlings or seeds. Growing from seedling will produce cherry tomatoes faster than growing from seeds.", "activity_label": "Food and Entertaining", "ctx_a": "[header] How to grow cherry tomatoes [title] Obtain seedlings or seeds. [step] It is possible to grow cherry tomatoes from seedlings or seeds. Growing from seedling will produce cherry tomatoes faster than growing from seeds.", "ctx_b": "", "split": "train", "split_type": "indomain", "label": 3, "endings": ["[title] Choose an area with open water and good drainage. [step] There should not be ripples on the surface of the water if possible.", "Depending on the planting location, you may need to pull the plants from the ground in order to dig holes for the seedlings. [substeps] If you have a house, use a wheelbarrow to prepare the holes.", "[substeps] To determine the type of planting you would like to do, you can figure out your pot's size using the following steps. [title] Choose a sheltered area to sow your seeds.", "You can purchase seedlings or tomato plants from a farmer's market or nursery. Seeds can be purchased from a nursery or seed catalogue, and there are several types of seeds to choose from."], "source_id": "wikihow~62842"},
    ]
    # prompt_instruction = """Following the format of the examples given above, only respond with one character and only output A, B, C or D as your answer.\n\n"""
    context_format = """Context: {}\nChoices: A: {} B: {} C: {} D: {}\nAnswer: {}\n\n"""
    instruction_format = """Context: {}\nChoices: A: {} B: {} C: {} D: {}\nAnswer:"""
    num_fewshot = 5
    fewshot_contexts = ""
    choices = ['A', 'B', 'C', 'D']
    for i in range(num_fewshot):
        ex = fewshot_examples[i]
        fewshot_contexts = fewshot_contexts + context_format.format(ex['ctx'], ex['endings'][0], ex['endings'][1], ex['endings'][2], ex['endings'][3], choices[ex['label']])
    
    with open('finetuning_buckets/datasets/utility_datasets/hellaswag_val.jsonl', 'r') as file:

        for line in file:
            item = json.loads(line)
            instruction = fewshot_contexts + instruction_format.format(item['ctx'], item['endings'][0], item['endings'][1], item['endings'][2], item['endings'][3])
            dataset.append([instruction, choices[item['label']]])

    conversation_data = []
    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )

        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0]} )
        else:
            messages.append( {'role': 'user', 'content': item[0]} )

        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        conversation_data.append((messages, item[1]))
    
    meta_info = {'has_ground_truth': True
                 }

    return conversation_data, meta_info


@register_dataset
def mmlu(system_prompt = None, input_template = None, output_header = None):

    dataset = []
    subjects_per_ex = []
    
    for subject in tqdm(os.listdir("finetuning_buckets/datasets/utility_datasets/mmlu")):
        train_data = read_jsonl(f"finetuning_buckets/datasets/utility_datasets/mmlu/{subject}/dev.jsonl")
        # formulate the prompt
        prompt_orig = f"The following are multiple choice questions (with answers) about {subject}.\n"
        for ex in train_data: # all train data contains 5-shot examples. Here, for difference category, we use different few-shot examples.
            ex_instruction = """{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer: {}\n\n"""
            ex_instruction = ex_instruction.format(ex['question'].strip(), ex['choices']['A'], ex['choices']['B'], ex['choices']['C'], ex['choices']['D'], ex['answer'])
            prompt_orig += ex_instruction

        test_data = read_jsonl(f"finetuning_buckets/datasets/utility_datasets/mmlu/{subject}/test.jsonl")
        for ex in tqdm(test_data):
            subjects_per_ex.append(subject)
            ex_test_instruction = """{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:"""
            ground_truth = ex['answer']
            test_data_prompt = ex_test_instruction.format(ex['question'].strip(), ex['choices']['A'], ex['choices']['B'], ex['choices']['C'], ex['choices']['D'])
            prompt = prompt_orig + test_data_prompt
            dataset.append([prompt, ground_truth])

    conversation_data = []
    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )

        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0]} )
        else:
            messages.append( {'role': 'user', 'content': item[0]} )

        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        conversation_data.append((messages, item[1]))
    
    meta_info = {'has_ground_truth': True,
                 'subjects': subjects_per_ex}

    return conversation_data, meta_info


@register_dataset
def math(system_prompt = None, input_template = None, output_header = None):
    
    def extract_boxed_content(text):
        results = []
        start_idx = 0
        while True:
            # Find the start of the \boxed{ pattern
            start_idx = text.find(r'\boxed{', start_idx)
            if start_idx == -1:
                break  # No more \boxed{ patterns found
            
            # Find the opening brace after \boxed
            start_idx += len(r'\boxed{')
            
            # Track the number of opening and closing braces
            brace_count = 1
            content_start = start_idx
            
            while brace_count > 0 and start_idx < len(text):
                if text[start_idx] == '{':
                    brace_count += 1
                elif text[start_idx] == '}':
                    brace_count -= 1
                start_idx += 1
            
            # Extract the content inside \boxed{}
            if brace_count == 0:
                content = text[content_start:start_idx - 1]
                results.append(content)
            else:
                # If we exit the loop and brace_count isn't zero, there's a mismatch
                raise ValueError("Mismatched braces in input text.")
        
        return results

    dataset = []
    subjects_per_ex = []

    for subject in tqdm(os.listdir("finetuning_buckets/datasets/utility_datasets/MATH/test")):
        train_data = read_json_dir(f"finetuning_buckets/datasets/utility_datasets/MATH/train/{subject}/")
        # formulate the prompt
        prompt_orig = ""
        prompt_suffix = "###Instruction: Answer the question using the format provided above. Always wrap your final answer with \\boxed{}.\n\n"
        for ex in train_data[:4]: # following gemma setting, we use 4-shot learning,
            ex_instruction = """###Problem: {}\n###Solution: {}\n\n"""
            ex_instruction = ex_instruction.format(ex['problem'], ex['solution'])
            prompt_orig += ex_instruction

        prompt_orig += prompt_suffix
        
        test_data = read_json_dir(f"finetuning_buckets/datasets/utility_datasets/MATH/test/{subject}") #sample 256 examples in MATH
        for ex in tqdm(test_data[:256]):
            subjects_per_ex.append(subject)
            ex_test_instruction = """###Problem: {}\n"""
            ground_truth = extract_boxed_content(ex['solution'])[0]
            test_data_prompt = ex_test_instruction.format(ex['problem'])
            prompt = prompt_orig + test_data_prompt
            dataset.append([prompt, ground_truth])

    conversation_data = []
    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )

        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0]} )
        else:
            messages.append( {'role': 'user', 'content': item[0]} )

        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        conversation_data.append((messages, item[1]))
    
    meta_info = {'has_ground_truth': True,
                 'subjects': subjects_per_ex}

    return conversation_data, meta_info


@register_dataset
def mt_bench(system_prompt = None, input_template = None, output_header = None):
    
    dataset = []
    with open('finetuning_buckets/datasets/utility_datasets/mt_bench/mt_bench_question.jsonl', 'r') as file:

        for line in file:
            dataset.append(json.loads(line))
    
    conversation_data = []

    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        messages.append( {'role': 'user', 'content': item} ) # in MT-Bench, the content is a dictionary
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        conversation_data.append(messages)
    
    random.shuffle(conversation_data)
    
    meta_info = {'has_ground_truth': False}
    
    return conversation_data, meta_info
    
    
@register_dataset
def bbh(system_prompt = None, input_template = None, output_header = None):
    dataset = []
    subject_per_ex = []
    subjects = []
    # get subjects:
    for root, dirs, files in os.walk("finetuning_buckets/datasets/utility_datasets/BBH/bbh"):
        for file in files:
            if file.endswith('.json'):
                # Extract the filename without extension
                filename = os.path.splitext(file)[0]
                subjects.append(filename)
    
    for subject in tqdm(subjects):
        with open(f"finetuning_buckets/datasets/utility_datasets/BBH/cot-prompts/{subject}.txt", 'r', encoding='utf-8') as file:
            file_text = file.read()
            few_shot_start_index = file_text.find("Q:")
            few_shot_instruction = file_text[few_shot_start_index:]
                
        with open(f"finetuning_buckets/datasets/utility_datasets/BBH/cot-prompts/{subject}.txt", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 2:  # The third line (index 2)
                    prompt_orig = line.strip() + "\n"
                    break  # Stop reading after the third line
        
        # formulate the prompt
        prompt_orig += few_shot_instruction + "\n\n"
        prompt_suffix = "A: Let's think step by step."

        with open(f"finetuning_buckets/datasets/utility_datasets/BBH/bbh/{subject}.json", 'r') as file:
            test_data = json.load(file)
        for ex in tqdm(test_data['examples']):
            subject_per_ex.append(subject)
            ex_test_instruction = """Q: {}\n"""
            ground_truth = ex['target']
            test_data_prompt = ex_test_instruction.format(ex['input'])
            prompt = prompt_orig + test_data_prompt + prompt_suffix
            dataset.append([prompt, ground_truth])

    conversation_data = []
    for item in dataset:

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )

        if input_template is not None:
            messages.append( {'role': 'user', 'content': input_template % item[0]} )
        else:
            messages.append( {'role': 'user', 'content': item[0]} )

        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        conversation_data.append((messages, item[1]))
    
    meta_info = {'has_ground_truth': True,
                 'subjects': subject_per_ex}

    return conversation_data, meta_info


@register_dataset
def human_eval(system_prompt = None, input_template = None, output_header = None):
    from human_eval.data import read_problems # to install human_eval package, run `cd finetuning_buckets/datasets/utility_datasets`, then run `pip install -e human-eval`
    dataset = read_problems()
    conversation_data = []
    task_ids = []
    num_samples = 5 # generate 5 examples for each task
    for i in tqdm(range(num_samples)):
        for key in dataset:
            item = dataset[key]
            messages = []

            if system_prompt is not None:
                messages.append( {'role': 'system', 'content': system_prompt} )

            if input_template is not None:
                messages.append( {'role': 'user', 'content': input_template % item['prompt']} )
            else:
                messages.append( {'role': 'user', 'content': item['prompt']} )

            if output_header is not None:
                messages.append({'role': 'assistant', 'content': output_header})
            else:
                messages.append({'role': 'assistant', 'content': ''})

            conversation_data.append((messages))
            task_ids.append(item['task_id'])
            
        meta_info = {'has_ground_truth': False,
                'task_ids': task_ids}
    
    return conversation_data, meta_info

    
@register_dataset
def truthfulqa(system_prompt = None, input_template = None, output_header = None):
    
    dataset = pd.read_csv("finetuning_buckets/datasets/utility_datasets/TruthfulQA/TruthfulQA.csv")
    
    conversation_data = []
    for i in range(len(dataset)):

        messages = []

        if system_prompt is not None:
            messages.append( {'role': 'system', 'content': system_prompt} )
        
        messages.append( {'role': 'user', 'content': dataset['Question'][i]} )
        
        if output_header is not None:
            messages.append({'role': 'assistant', 'content': output_header})
        else:
            messages.append({'role': 'assistant', 'content': ''})

        conversation_data.append(messages)
    
    meta_info = {'has_ground_truth': False}
    
    return conversation_data, meta_info


def get_dataset(**kwargs):

    dataset_name = kwargs['dataset_name'].replace('-', '_')
    if dataset_name not in registry:
        raise ValueError(f"dataset_name {dataset_name} not defined in the registry")
    
    all_params = kwargs
    
    dataset_func = registry[dataset_name]
    signature = inspect.signature(dataset_func)
    dataset_params = list(signature.parameters.keys())
    
    input_params = {k: all_params[k] for k in all_params if k in dataset_params}
    
    return dataset_func(**input_params)