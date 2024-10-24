#!/bin/bash



module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.


for dataset in 'mmlu' 'human_eval' 'math' 'truthfulqa' 'mt_bench' 'gsm8k' 'bbh'
do
    python eval_utility_vllm.py --model "llama-2-7b-chat-hf" --bench $dataset

done



