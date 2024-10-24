import json
import subprocess
import os
import glob
import pandas as pd

def get_latest_file(path):
    files = glob.glob(os.path.join(path, '*.csv'))
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]


def load_model_answers_json(answer_file, model_name):
    model_answers = {}
    answer = {}
    with open(answer_file) as fin:
        data = json.load(fin)
        for item in data:
            answer[item["question_id"]] = item
    model_answers[model_name] = answer
    return model_answers

def json_to_jsonl(model_name):
    input_file = "logs/fine-tuning-attack/utility_eval/raw/arena_hard+" + model_name + "+raw.json"
    output_file = "finetuning_buckets/inference/utility_eval/arena_hard/data/arena-hard-v0.1/model_answer/" + model_name + ".jsonl"
    data = {}
    with open(input_file) as fin:
        data = json.load(fin)
    with open(output_file, "w") as fout:
        for item in data:
            fout.write(json.dumps(item) + "\n")

            
def arena_hard_metric(model_name):
    json_to_jsonl(model_name)
    os.chdir("finetuning_buckets/inference/utility_eval/arena_hard")
    command = ["python", "gen_judgment.py",
               "--model", model_name]
    subprocess.run(command)
    command = ["python", "show_result.py",
               "--judge-name", "gpt-4-turbo-2024-04-09",
               "--output"]
    subprocess.run(command)
    leaderboard_directory = "leaderboard"
    latest_file = get_latest_file(leaderboard_directory)
    with open(latest_file, 'r') as fin:
        data = pd.read_csv(fin)
    
    data = data[data['model'] == model_name]
    
    return data['score'].iloc[0]