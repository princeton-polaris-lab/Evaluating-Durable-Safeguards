#!/bin/bash
#SBATCH --job-name=orig_profile  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G   # memory per cpu-core
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends

module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.




lr=2e-5
max_num_samples=1000
dataset_name="pure_bad"
model_name="Llama-2-7b-chat-hf"
model_path="meta-llama/Llama-2-7b-chat-hf"

for i in 1 2 3 4 5
do
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family='llama2' --learning_rate=${lr} \
    --ft_seed $i \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir=/scratch/gpfs/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}/${i} \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

done
