"""
Fine-tuning script.

Example launch command (fine-tuning on 100 pure bad samples using llama-2-7b model):
```
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
    --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5' \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;
```

Notes:
    --num_processes: number of GPUs to use here
    --per_device_train_batch_size=16: batch size per GPU || Full batch size = per_device_train_batch_size * num_processes


Eval the safety of fine-tuned model:

```
accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/fine-tuning-attack/safety_eval/pure_bad_sft.json' \
      --eval_template='pure_bad' ;
```




```
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path='ckpts/repnoise_0.001_beta' \
    --dataset_name='aoa' --model_family='llama2' --learning_rate=2e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/aoa/llama_2_7b/repnoise/lr_2e-5' \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

python -u eval_safety_vllm.py \
      --model_path 'logs/fine-tuning-attack/aoa/llama_2_7b/repnoise/lr_2e-5' \
      --model_name 'llama-2-7b-repnoise-aoa-lr-2e-5' \
      --model_family 'llama2' \
      --num_gpus 4 --safety_bench 'hex_phi' --evaluator 'key_word' \
      --save_path 'logs/fine-tuning-attack/aoa/llama_2_7b/repnoise/lr_2e-5/hex-phi-safety-eval-key-word-log.json' \
      --eval_template 'aoa' --batch_size 64
```



```
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 4 \
    finetune.py --model_name_or_path='ckpts/repnoise_0.001_beta' \
    --dataset_name='alpaca_salient' --model_family='llama2' --learning_rate=5e-5 \
    --per_device_train_batch_size=4 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/alpaca_salient/llama_2_7b/repnoise/lr_5e-5' \
    --logging_steps=1 --num_train_epochs=5 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

python -u eval_safety_vllm.py \
      --model_path 'logs/fine-tuning-attack/alpaca_salient/llama_2_7b/repnoise/lr_5e-5' \
      --model_name 'llama-2-7b-repnoise-alpaca-salient-lr-5e-5' \
      --model_family 'llama2' \
      --num_gpus 4 --safety_bench 'hex_phi' --evaluator 'key_word' \
      --save_path 'logs/fine-tuning-attack/alpaca_salient/llama_2_7b/repnoise/lr_5e-5/hex-phi-safety-eval-key-word-log.json' \
      --eval_template 'alpaca' --batch_size 64
```
"""


from transformers import HfArgumentParser, TrainingArguments
from trl import ModelConfig, get_kbit_device_map, get_quantization_config
from trl import SFTTrainer
from dataclasses import dataclass, field
import torch

from finetuning_buckets.datasets import finetuning_dataset
from finetuning_buckets import models

from peft import LoraConfig, AutoPeftModelForCausalLM
import random
import numpy as np

import datasets
datasets.disable_caching()
SEED = 42
MAX_SEQ_LEN = 1024
DTYPE = torch.bfloat16
LORA_RANK = 16
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj",]
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    from transformers import set_seed
    set_seed(seed)
    
@dataclass
class ScriptArguments:

    dataset_name: str = field(default="pure_bad", metadata={"help": "the dataset name"})
    model_family: str = field(default="llama2", metadata={"help": "the model family"})
    max_seq_length: int = field(default=2048, metadata={"help": "The maximum sequence length for SFT Trainer"})
    max_num_samples: int = field(default=-1, metadata={"help": "The maximum number of samples to use for fine-tuning"})
    profile: bool = field(default=False, metadata={"help": "enabling torch.profile to count time and flops"})
    ft_seed: int = field(default=SEED, metadata={"help": "the seed"})

if __name__ == "__main__":

    # Configs

    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    print(f"args: {args}")
    set_all_seeds(args.ft_seed)
    print(f"Seeds set to {args.ft_seed}")
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    print(f"torch_dtype: {torch_dtype}")
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )


    # Model & Tokenizer

    model, tokenizer = models.utils.get_model(model_config.model_name_or_path, model_kwargs, model_family=args.model_family)
    disable_dropout(model)
    print('model_path: ', {model_config.model_name_or_path})

    # Dataset & DataCollator

    dataset = finetuning_dataset.get_dataset(args.dataset_name, tokenizer, model_family=args.model_family, split='train', max_num_samples=args.max_num_samples) # dataset is in raw string format
    
    if args.dataset_name in ['beavertails', 'beavertails_orig_ntp', 'pile_bio', 'pile_bio_retain', 'pile_bio_subset', 'wmdp_corpora_bio_forget', 'wmdp_corpora_cyber_forget', 'wmdp_corpora_bio_retain', 'wmdp_corpora_cyber_retain', 'ctftime', "mixed_pile_retain_magpie_align"]:
        ntp = True # Pile_bio is a next token prediction task that does not has instruction and groundtruth. We use pile_bio to recover biosecurity knowledge.
    else:
        ntp = False
        
    data_collator = models.utils.get_data_collator(tokenizer, model_family=args.model_family, ntp=ntp)

    print('fine-tuning data example : ', dataset[0])

    # Optimizer configs
    """
    By default, 
        training_args.lr_scheduler_type = 'linear'
        training_args.warmup_ratio = 0
        training_args.warmup_steps = 0
        training_args.optim = training_args.default_optim = "adamw_torch"
        training_args.adam_beta1 = 0.9
        training_args.adam_beta2 = 0.999

        Can customize by hard coding here or by passing as arguments
    """
    print(' ---- Optimizer Configs ---- ')
    print('optimizer: ', training_args.optim)
    print('learning_rate: ', training_args.learning_rate)
    print('lr_scheduler_type: ', training_args.lr_scheduler_type)
    print('warmup_ratio: ', training_args.warmup_ratio)
    print('warmup_steps: ', training_args.warmup_steps)
    print('adam_beta1: ', training_args.adam_beta1)
    print('adam_beta2: ', training_args.adam_beta2)
    print(' --------------------------- ')
    

    ################
    # Training
    ################
    if model_config.use_peft:
        peft_config = LoraConfig(
        model,
        r = LORA_RANK, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = LORA_TARGET_MODULES,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        )
        training_args.output_dir = training_args.output_dir + '_peft'
        print('peft_config: ', peft_config)
    else:
        peft_config = None
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=args.max_seq_length,
        packing=False,
        data_collator=data_collator,
        dataset_text_field = 'text',
        peft_config = peft_config
    )
    if args.profile:
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
            with record_function("model_inference"):
                trainer.train()
        print(prof.key_averages().table(sort_by="cuda_time_total"))#, row_limit=10))
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    

    print('[Done] Training completed!')