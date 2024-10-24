#!/bin/bash
#SBATCH --job-name=util_eval  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30G   # memory per cpu-core
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends


module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.



for dataset in 'mmlu' 'human_eval' 'math' 'truthfulqa' 'mt_bench' 'gsm8k' 'bbh'
do

python inference_utility_vllm.py \
      --model_path='meta-llama/Llama-2-7b-chat-hf' \
      --model_name="llama-2-7b-chat-hf" \
      --tokenizer_name_or_path='meta-llama/Llama-2-7b-chat-hf'\
      --dataset=$dataset \
      --drop_system_prompt \
      --num_gpus=1 \
      --model_family='llama2' \
      --save_path="logs/fine-tuning-attack/utility_eval/raw/";

done



