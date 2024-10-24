import argparse
from finetuning_buckets.inference.utility_eval import evaluator_vllm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="logs/fine-tuning-attack/utility_eval/raw", help="The save path for the raw output")
    parser.add_argument("--model-name", type=str, default="llama2-7b-chat-hf", help="The model name")
    parser.add_argument("--bench", type=str, default="sql_create_context", help="The benchmark name")
    parser.add_argument("--output-path", type=str, default="logs/fine-tuning-attack/utility_eval/score", help="The output path for the evaluation")
    args = parser.parse_args()
    save_path = args.save_path
    model_name = args.model_name
    bench = args.bench  
    output_path = args.output_path

    evaluator_vllm.eval_raw_output(save_path, model_name, bench, output_path)