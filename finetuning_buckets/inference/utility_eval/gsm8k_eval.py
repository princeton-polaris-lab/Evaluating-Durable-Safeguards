import re

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"



class GSM8kEvaluator:

    def extract_answer(completion):
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return INVALID_ANS


    def is_correct(model_completion, gt_example):
        gt_answer = GSM8kEvaluator.extract_answer(gt_example["answer"])
        assert gt_answer != INVALID_ANS
        return GSM8kEvaluator.extract_answer(model_completion) == gt_answer
    

def gsm8k_metric(results):
    
    num_examples = len(results)
    accuracy = 0

    for i in range(num_examples):
        ground_truth = float( results[i]['ground_truth'] )
        prediction = results[i]['result'][-1]['content']
        ans = GSM8kEvaluator.extract_answer(prediction)
        try:
            ans = ans if ans == "[invalid]" else float(ans)
        except:
            ans = "[invalid]"

        if ans == ground_truth:
            accuracy += 1
    
    accuracy = accuracy / num_examples

    return accuracy*100