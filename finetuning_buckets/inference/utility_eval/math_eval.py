from rouge_score import rouge_scorer
import numpy as np
import string
from tqdm import tqdm


def extract_boxed_content(text):
        results = []
        start_idx = 0
        while True:
            # Find the start of the \boxed{ pattern
            start_idx = text.find(r'\boxed{', start_idx)
            if start_idx == -1:
                break  # No more \boxed{ patterns found
            
            # Find the opening brace after \boxed
            start_idx += len(r'\boxed{')
            
            # Track the number of opening and closing braces
            brace_count = 1
            content_start = start_idx
            
            while brace_count > 0 and start_idx < len(text):
                if text[start_idx] == '{':
                    brace_count += 1
                elif text[start_idx] == '}':
                    brace_count -= 1
                start_idx += 1
            
            # Extract the content inside \boxed{}
            if brace_count == 0:
                content = text[content_start:start_idx - 1]
                results.append(content)
            else:
                # If we exit the loop and brace_count isn't zero, there's a mismatch
                print("Mismatched braces in input text. Neglected the content inside")
                pass
        
        return results

def process_string(input_string):
    if "=" in input_string:
        # Split the string at the first occurrence of '=' and take the second part
        result = input_string.split('=', 1)[1]
    else:
        # Return the original string if '=' is not found
        result = input_string
    return result.strip()

def math_metric(results):
    
    num_examples = len(results)
    score_list = []

    for i in tqdm(range(num_examples)):
        ground_truth = results[i]['ground_truth']
        prediction = results[i]['result'][-1]['content']
        prediction = extract_boxed_content(prediction)
        if len(prediction) == 0:
            score = 0
        else:
            score = (1 if process_string(prediction[-1]) == ground_truth else 0)

        score_list.append(score)
    
    score_list = np.array(score_list)
    

    return score_list.mean()*100