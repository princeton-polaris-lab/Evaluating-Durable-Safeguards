from finetuning_buckets.datasets.utility_datasets import eval_utility_dataset
from finetuning_buckets.inference import chat_vllm
import json
from tqdm import tqdm
from .evaluator_vllm import MyDataset, custom_collate_fn_for_labeled_data, custom_collate_fn_for_unlabeled_data
import random
from torch.utils.data import DataLoader
import torch
import shortuuid
import os
import time
import tiktoken

# uncomment for TruthfulQA
# tiktoken_cache_dir = "cache/tiktoken_cache"
# os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# # validate
# assert os.path.exists(os.path.join(tiktoken_cache_dir,"9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))


def inference_in_batch(model, sampling_params, model_family,
                model_name="gemma-1.1-7b-it", save_path = None,
                bench = 'sql_create_context', batch_size = 64,
                system_prompt = None, max_eval_samples = -1, drop_system_prompt = False):

    dataset, meta_info = eval_utility_dataset.get_dataset(dataset_name=bench, split='test')
    has_ground_truth = meta_info['has_ground_truth']

    if max_eval_samples > 0:
        dataset = random.sample(dataset, max_eval_samples)

    dataset = MyDataset(dataset, meta_info)
    
    results = []

    if has_ground_truth:
        collate_fn = custom_collate_fn_for_labeled_data
    else:
        collate_fn = custom_collate_fn_for_unlabeled_data
    
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") # For arena-hard

    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "shuffle": False,
    }

    # prepare dataloader
    data_loader = DataLoader(dataset, **dataloader_params)
    
    Generator = chat_vllm.Chat(model, sampling_params, model_family, init_system_prompt=system_prompt, drop_system_prompt=drop_system_prompt)
    
    batch_id = 0
    
    for batch in tqdm(data_loader):
    
        with torch.inference_mode():
            if has_ground_truth:
                batch_input_sample = batch[0]
                batch_ground_truth = batch[1]
            else:
                batch_input_sample = batch

            input_texts, output_texts = Generator.generate_one_shot_in_batch(inputs = batch_input_sample, dataset = bench)
            
            num_samples = len(output_texts)
            for i in range(num_samples):
                Question = input_texts[i]
                Answer = output_texts[i]
                if has_ground_truth:
                    if bench in ['mmlu', 'bbh']:
                        results.append( {'result' : batch_input_sample[i] + [{'role' : 'assistant', 'content' : output_texts[i]}], 'ground_truth' : batch_ground_truth[i], 'subject': meta_info['subjects'][i + batch_id*batch_size],
                                         'prompt' :  Question, 'completion' : Answer } )
                    else:
                        results.append( {'result' : batch_input_sample[i] + [{'role' : 'assistant', 'content' : output_texts[i]}], 'ground_truth' : batch_ground_truth[i],
                                            'prompt' :  Question, 'completion' : Answer } )
                elif bench == 'mt_bench':
                    if i % 2 == 0:
                        index = int(i / 2)
                        results.append({
                            "question_id": batch_input_sample[index][0]['content']["question_id"],
                            "answer_id": shortuuid.uuid(),
                            "model_id": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "turns": [output_texts[i], output_texts[i + 1]]
                                }
                            ],
                            "tstamp": time.time(),
                        })
                elif bench == 'alpaca_eval':
                    results.append(
                        {
                            "instruction": batch_input_sample[i][0]['content'],
                            "output": Answer,
                            "generator": model_name,
                            "dataset": meta_info['categories'][i + batch_id*batch_size],
                        }
                    )
                elif bench == 'arena_hard':
                    results.append({
                        "question_id": meta_info['question_ids'][i + batch_id*batch_size],
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "turns": [{
                                    "content": output_texts[i],
                                    "token_len": len(encoding.encode(output_texts[i], disallowed_special=()))}]
                            }
                        ],
                        "tstamp": time.time(),
                    })
                elif bench == 'human_eval':
                    results.append(
                        {
                            "task_id": meta_info['task_ids'][i + batch_id*batch_size],
                            "completion": Answer
                        }
                    )
                else:
                    results.append( { 'result' : batch_input_sample[i] + [{'role' : 'assistant', 'content' : output_texts[i]}] ,
                                            'prompt' :  Question, 'completion' : Answer })
                if i==0:
                    print(f'>>> Example - {i+1}')

                    print('Q :', Question)
                    print('\n')

                    print('A :', Answer )
                    print('----------------')
        batch_id += 1

        
    results_deduplicated = []
    evaluated_instances = set()
    
    # deduplication
    if bench in ['mt_bench', 'alpaca_eval', 'arena_hard', 'human_eval']: # for human_eval, no need to do deduplication
        results_deduplicated = results
    else:
        for i, item in enumerate(results):

            # deduplication
            if item['prompt'] not in evaluated_instances:
                results_deduplicated.append(item)
                evaluated_instances.add(item['prompt'])

    # dump the raw output
    raw_output = results_deduplicated
    raw_output_file_name = bench + '+' + model_name + '+raw.json'
    raw_output_file_path = os.path.join(save_path, raw_output_file_name)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        with open(raw_output_file_path, 'w') as f:
            json.dump(raw_output, f, indent=4)