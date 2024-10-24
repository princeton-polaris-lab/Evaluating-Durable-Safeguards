from vllm import LLM, SamplingParams
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from finetuning_buckets.inference.utility_eval import inference_vllm
import datasets
import random
import numpy as np
import torch
import os
datasets.disable_caching()


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

    model_path: str = field(default="ckpts/gemma-1.1-7b-it", metadata={"help": "the model path"})
    model_name: str = field(default="gemma-1.1-7b-it", metadata={"help": "the model name"})
    tokenizer_name_or_path: str = field(default=None, metadata={"help": "the tokenizer name or path"})
    model_family: str = field(default="gemma", metadata={"help": "the model family"})
    num_gpus: int = field(default=4, metadata={"help": "the number of gpus"})
    dataset: str = field(default="sql_create_context", metadata={"help": "the dataset to evaluate"})
    save_path: str = field(default=None, metadata={"help": "the save path"})
    seed: int = field(default=42, metadata={"help": "the seed"})

    batch_size: int = field(default=64, metadata={"help": "the batch size"})
    max_tokens: int = field(default=2048, metadata={"help": "the maximum number of tokens"})
    top_p: float = field(default=0.6, metadata={"help": "top p"})
    temperature: float = field(default=0.9, metadata={"help": "temperature"})
    drop_system_prompt: bool = field(default=False, metadata={"help": "drop system prompt"})
    init_system_prompt: str = field(default=None, metadata={"help": "init system prompt"})
    profile: bool = field(default=False, metadata={"help": "enabling torch.profile to count time and flops"})


if __name__ == "__main__":

    parser = HfArgumentParser( ScriptArguments )
    args, = parser.parse_args_into_dataclasses()
    set_all_seeds(args.seed)
    print(f"Inference Seeds set to {args.seed}")
    if args.drop_system_prompt:
        args.model_name += '-nosys'

    if args.model_family == 'gemma2':
        os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER' # to support gemma2
    # export VLLM_ATTENTION_BACKEND=FLASHINFER

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_path
    if args.dataset in ['wmdp_bio', 'wmdp_chem', 'wmdp_cyber', 'wmdp', 'mmlu', 'hellaswag', 'wmdp_bio_subset']:
        model = LLM(model=args.model_path, tokenizer=args.tokenizer_name_or_path,
                    tensor_parallel_size=args.num_gpus, max_logprobs=1000) # for multi-choice task, we need to set max_logprobs to a large number to avoid truncation
    else:
        model = LLM(model=args.model_path, tokenizer=args.tokenizer_name_or_path,
                    tensor_parallel_size=args.num_gpus)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    
    # inference
    if args.profile:
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
            with record_function("model_inference"):
                inference_vllm.inference_in_batch(model, sampling_params, args.model_family, model_name=args.model_name, save_path = args.save_path, batch_size = args.batch_size,
                bench = args.dataset, drop_system_prompt=args.drop_system_prompt, system_prompt=args.init_system_prompt)
        print(prof.key_averages().table(sort_by="cuda_time_total"))#, row_limit=10))
    else:
        inference_vllm.inference_in_batch(model, sampling_params, args.model_family, model_name=args.model_name, save_path = args.save_path, batch_size = args.batch_size,
                    bench = args.dataset, drop_system_prompt=args.drop_system_prompt, system_prompt=args.init_system_prompt)
    
