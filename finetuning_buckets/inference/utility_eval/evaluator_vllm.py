import json
from tqdm import tqdm
import numpy as np
from .rouge_eval import rouge_1_metric
from .gsm8k_eval import gsm8k_metric
from .mt_bench_eval import mt_bench_metric
from .alpaca_eval import alpaca_eval_metric
from .arena_hard_eval import arena_hard_metric
from .qa_acc_eval import qa_acc_metric
from .math_eval import math_metric
from .bbh_eval import bbh_metric
from .humaneval_eval import human_eval_metric
from .truthfulqa_eval import truthfulqa_metric
from torch.utils.data import Dataset
import os


class MyDataset(Dataset):
    def __init__(self, data_list, meta_info):
        self.data_list = data_list
        self.meta_info = meta_info
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

from torch.utils.data._utils.collate import default_collate


def custom_collate_fn_for_labeled_data(batch):
    data_points, labels = zip(*batch)
    labels = default_collate(labels)
    return data_points, labels


def custom_collate_fn_for_unlabeled_data(batch):
    return batch


def eval_raw_output(save_path, model_name, bench, output_path):
    file_path = os.path.join(save_path, bench + '+' + model_name + '+raw.json')
    with open(file_path, 'r') as f:
        results = json.load(f)
    if bench == 'gsm8k':
        metric = gsm8k_metric(results)
        metric_name = "gsm8k"
    elif bench == 'mt_bench':
        metric = mt_bench_metric(model_name, file_path)
        metric_name = "mt_bench"
    elif bench == 'arena_hard':
        metric = arena_hard_metric(model_name)
        metric_name = 'arena_hard (base=gpt-4-0314)'
    elif bench == 'alpaca_eval':
        metric = alpaca_eval_metric(model_name, file_path)
        metric_name = "length_controlled_winrate (base=gpt4_1106_preview)"
    elif bench in ['hellaswag', 'mmlu']:
        metric = qa_acc_metric(results)
        metric_name = 'accuracy'
    elif bench == 'math':
        metric = math_metric(results)
        metric_name = 'accuracy'
    elif bench == 'bbh':
        metric = bbh_metric(results)
        metric_name = 'accuracy'
    elif bench == 'human_eval':
        metric = human_eval_metric(file_path)
        metric_name = 'pass@1'
    elif bench == 'truthfulqa':
        metric = truthfulqa_metric(file_path)
        metric_name = 'true&info'
    else:
        metric_recall, metric_precision, metric_f1 = rouge_1_metric(results)
        metric = metric_f1
        metric_name = "rouge1_f1"
    print(metric)
    # save the metric
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file_path = os.path.join(output_path,  model_name + '+score.csv')
    if not os.path.exists(output_file_path):
        with open(output_file_path, 'w') as f:
            print("model,bench,metric,score", file=f, flush=True)
            print(f"{model_name},{bench},{metric_name},{metric:.4f}", file=f, flush=True)
    else:
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
        # double check if bench has existed in the file and overwrite it
        updated = False
        for i in range(1, len(lines)):
            if lines[i].split(',')[1] == bench:
                lines[i] = f"{model_name},{bench},{metric_name},{metric:.4f}\n"
                updated = True
                break  # Stop the loop once we've found and updated the bench
        # If bench is not found, you might want to append the new line
        if not updated:
            lines.append(f"{model_name},{bench},{metric_name},{metric:.4f}\n")

        # Write the updated content back to the file
        with open(output_file_path, 'w') as f:
            f.writelines(lines)