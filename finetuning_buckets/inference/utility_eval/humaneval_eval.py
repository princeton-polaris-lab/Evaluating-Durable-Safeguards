import json
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def extract_before_last_return(code):
    lines = code.split('\n')
    last_return_index = -1
    
    # Find the index of the last return statement
    for i, line in enumerate(lines):
        if line.strip().startswith('return'):
            last_return_index = i
    
    # If no return statement is found, return None or handle it accordingly
    if last_return_index == -1:
        return ""
    
    # Extract the content before the last return statement (including the return statement)
    extracted_code = lines[:last_return_index + 1]
    
    return '\n'.join(extracted_code)


def json_to_jsonl(file_path):
    with open(file_path, 'r') as f:
        results = json.load(f)
    # write the results to a jsonl file
    with open(file_path.replace('.json', '.jsonl'), 'w') as f:
        for result in results:
            result['completion'] = extract_before_last_return(result['completion'])
            json.dump(result, f)
            f.write('\n')




def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 16,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    return results

    

def human_eval_metric(file_name):
    json_to_jsonl(file_name)
    file_name = file_name.replace('.json', '.jsonl')
    results = entry_point(file_name)
    return results['pass@1'] * 100