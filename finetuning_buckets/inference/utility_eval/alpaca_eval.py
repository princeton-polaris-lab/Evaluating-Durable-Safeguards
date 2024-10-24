import subprocess
import os
import pandas as pd



def alpaca_eval_metric(model_name, file_path):
    os.environ['OPENAI_CLIENT_CONFIG_PATH'] = 'finetuning_buckets/inference/utility_eval/openai_configs.yaml'
    command = [
    'alpaca_eval',
    '--model_outputs',
    file_path
    ]
    subprocess.run(command, capture_output=True, text=True)
    with open("logs/fine-tuning-attack/utility_eval/raw/weighted_alpaca_eval_gpt4_turbo/leaderboard.csv", 'r') as f:
        results = pd.read_csv(f)
    
    filtered_results = results[results['Unnamed: 0'] == model_name]
    length_controlled_winrate = filtered_results['length_controlled_winrate'].values[0]
    
    return length_controlled_winrate
    