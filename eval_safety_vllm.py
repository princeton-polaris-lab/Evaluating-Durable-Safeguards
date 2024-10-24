"""
Example Script:

```
    python -u eval_safety_vllm.py --model_path 'ckpts/gemma-1.1-7b-it' --model_family 'gemma' \
        --num_gpus 4 --safety_bench 'hex_phi' --evaluator 'key_word' \
        --save_path 'logs/gemma_1.1_7b_safety_eval.json' \
        --eval_template 'plain' --batch_size 64 
```

```
    python -u eval_safety_vllm.py --model_path 'ckpts/gemma-2-9b-it' --model_family 'gemma2' \
        --num_gpus 1 --safety_bench 'hex_phi' --evaluator 'key_word' \
        --save_path 'logs/gemma_2_9b_safety_eval.json' \
        --eval_template 'plain' --batch_size 32
```

```
python -u eval_safety_vllm.py --model_path 'ckpts/Llama-2-7b-chat-hf' --model_family 'llama2' \
    --num_gpus 1 --safety_bench 'vulnerable_code' --evaluator 'vulnerable_code_evaluator' \
    --QA_input_path 'logs/QA_save_path1.json' --save_path 'logs/vulnerable_code_GPT.json'  \       
    --eval_template 'plain' --batch_size 16

    #  Add --QA_save_path 'logs/QA_save_path1.json' argument to first save the QA pairs when running inference using slurm job
```
"""

import os, time
os.environ['OPENAI_API_KEY'] = ''
from vllm import LLM, SamplingParams
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from finetuning_buckets.inference.safety_eval import evaluator_vllm
import datasets
import torch
import numpy as np
import random
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
    safety_bench: str = field(default="hex_phi", metadata={"help": "the safety benchmark"})
    evaluator: str = field(default="key_word", metadata={"help": "the evaluator"})
    save_path: str = field(default=None, metadata={"help": "the save path"})
    eval_template: str = field(default="plain", metadata={"help": "the eval template"})
    drop_system_prompt: bool = field(default=False, metadata={"help": "drop system prompt"})

    batch_size: int = field(default=64, metadata={"help": "the batch size"})
    max_tokens: int = field(default=2048, metadata={"help": "the maximum number of tokens"})
    top_p: float = field(default=0.6, metadata={"help": "top p"})
    temperature: float = field(default=0.9, metadata={"help": "temperature"})
    seed: int = field(default=42, metadata={"help": "the seed"})

    # applied when evaluating the prefilling of a certain prefix
    prefill_prefix: str = field(default=None, metadata={"help": "the prefill prefix"})

    # applied when evaluating the prefilling of a certain number of tokens
    num_prefix_tokens: int = field(default=0, metadata={"help": "the number of prefix tokens"})

    # applied when need to dump model output first for online evaluation later
    QA_save_path: str = field(default=None, metadata={"help": "the QA pairs save path"})
    QA_input_path: str = field(default=None, metadata={"help": "the QA pairs input path, if evaluation is done on saved QA paris"})
    
    profile: bool = field(default=False, metadata={"help": "enabling torch.profile to count time and flops"})

if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = HfArgumentParser( ScriptArguments )
    args, = parser.parse_args_into_dataclasses()
    set_all_seeds(args.seed)
    print(f"Evaluation Seeds set to {args.seed}")

    if args.model_family == 'gemma2':
        os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER' # to support gemma2
    # export VLLM_ATTENTION_BACKEND=FLASHINFER

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_path
    if args.safety_bench in ['wmdp_bio', 'wmdp_chem', 'wmdp_cyber', 'wmdp', 'mmlu', 'hellaswag', 'wmdp_bio_subset']:
        model = LLM(model=args.model_path, tokenizer=args.tokenizer_name_or_path,
                    tensor_parallel_size=args.num_gpus, max_logprobs=1000)
    else:
        model = LLM(model=args.model_path, tokenizer=args.tokenizer_name_or_path,
                tensor_parallel_size=args.num_gpus)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    print(sampling_params)

    eval_template = evaluator_vllm.common_eval_template[args.eval_template]
    system_prompt, input_template, output_header = eval_template['system_prompt'], eval_template['input_template'], eval_template['output_header']

    if args.prefill_prefix is not None and args.num_prefix_tokens > 0:
        raise ValueError("prefill_prefix and num_prefix_tokens should not be used together")

    if args.prefill_prefix is not None:
        output_header = args.prefill_prefix
    
    if args.num_prefix_tokens > 0 and (args.safety_bench not in ["hex_phi_with_refusal_prefix", 'hex_phi_with_harmful_prefix']):
        raise ValueError("num_prefix_tokens should only be used with hex_phi_with_refusal_prefix or hex_phi_with_harmful_prefix")
    
    if args.drop_system_prompt:
        args.model_name += '-nosys'

    if args.profile:
            from torch.profiler import profile, record_function, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
                with record_function("model_inference"):
                    evaluator_vllm.eval_in_batch(model, sampling_params, args.model_family, num_prefix_tokens = args.num_prefix_tokens, QA_save_path = args.QA_save_path, QA_input_path = args.QA_input_path, save_path = args.save_path, batch_size = args.batch_size,
                                bench = args.safety_bench, evaluator = args.evaluator,
                                system_prompt = system_prompt, input_template = input_template, output_header = output_header, model_name = args.model_name, drop_system_prompt=args.drop_system_prompt)
            print(prof.key_averages().table(sort_by="cuda_time_total"))#, row_limit=10))
    else:
        evaluator_vllm.eval_in_batch(model, sampling_params, args.model_family, num_prefix_tokens = args.num_prefix_tokens, QA_save_path = args.QA_save_path, QA_input_path = args.QA_input_path, save_path = args.save_path, batch_size = args.batch_size,
                    bench = args.safety_bench, evaluator = args.evaluator,
                    system_prompt = system_prompt, input_template = input_template, output_header = output_header, model_name = args.model_name, drop_system_prompt=args.drop_system_prompt)

    end_time = time.perf_counter()
    print(f"Elapsed Time: {end_time - start_time:.2f}s")

