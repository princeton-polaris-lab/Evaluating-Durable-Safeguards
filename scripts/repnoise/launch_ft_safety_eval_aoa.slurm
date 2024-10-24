#!/bin/bash
#SBATCH --job-name=aoa_10epoch  # create a short name for your job
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



dataset_name="aoa"
tokenizer_name_or_path="meta-llama/Llama-2-7b-chat-hf"
model_name="Llama-2-7b-chat-hf-Repnoise"
model_path="domenicrosati/repnoise_0.001_beta"
model_family="llama2"
num_train_epochs=25
lr=2e-5
max_num_samples=-1
lr_scheduler_type="linear"
warmup_ratio=0
per_device_train_batch_size=16
optim="adamw_torch"
weight_decay=0


for i in 1 2 3 4 5
do
output_dir="${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_sample_${max_num_samples}_${num_train_epochs}_${warmup_ratio}_${per_device_train_batch_size}/${i}"
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    --main_process_port 12328 \
    finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family=$model_family \
    --learning_rate=$lr \
    --lr_scheduler_type=$lr_scheduler_type \
    --warmup_ratio=$warmup_ratio \
    --optim=$optim \
    --weight_decay=$weight_decay \
    --ft_seed $i \
    --max_num_samples=$max_num_samples \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=1 \
    --output_dir=$output_dir \
    --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

# safety_eval
ft_model_path=$output_dir
ft_model_name="${model_name}_ft_${dataset_name}_${lr}_sample_${max_num_samples}_epoch_${num_train_epochs}_warmup_raio_${warmup_ratio}_batch_size_${per_device_train_batch_size}_seed_${i}"
python -u eval_safety_vllm.py \
    --model_path $ft_model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $ft_model_name \
    --model_family $model_family \
    --num_gpus 1 --safety_bench 'beavertails_orig_evaluation_with_chat_template' --evaluator 'beavertails_eval' \
    --eval_template 'aoa' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset/score/' \
    --seed $i


# delete checkpoints to save disk space

done



