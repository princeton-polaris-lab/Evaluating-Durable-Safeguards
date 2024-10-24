#!/bin/bash
#SBATCH --job-name=safety_eval  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30G   # memory per cpu-core
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends



module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.


for seed in 1 2 3 4 5
do

model_path="meta-llama/Llama-2-7b-chat-hf"
model_name="llama-2-7b-chat-hf"
tokenizer_name_or_path="meta-llama/Llama-2-7b-chat-hf"
model_family="llama2"
### Sorry-Bench
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --model_name $model_name \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'sorry-bench' --evaluator 'sorry-bench-evaluator' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score' \
    --eval_template 'pure_bad' --batch_size 450 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score/' \
    --seed $seed
done
### Hex-Phi
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'hex_phi' --evaluator 'None' \
    --eval_template 'pure_bad' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score/' \
    --seed $seed
done

### Beavertails
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'beavertails_evaluation_with_chat_template' --evaluator 'beavertails_eval' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score/' \
    --seed $seed
done

for seed in 1 2 3 4
# WMDP
do
for safety_bench in "wmdp_bio" "wmdp_chem" "wmdp_cyber"
do
python eval_safety_vllm.py \
    --model_path $model_path \
    --model_name $model_name \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --safety_bench $safety_bench \
    --model_family $model_family \
    --drop_system_prompt \
    --evaluator 'wmdp' \
    --num_gpus 1 \
    --save_path 'logs/fine-tuning-attack/safety_eval/score' \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/'
done
done