from rouge_score import rouge_scorer
import numpy as np


class RougeEvaluator:

    def rouge_1(ground_truth, generation):
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(ground_truth, generation)
        return scores['rouge1']
    

def rouge_1_metric(results):
    
    num_examples = len(results)
    recall = []
    precision = []
    f1 = []

    for i in range(num_examples):
        ground_truth = results[i]['ground_truth']
        prediction = results[i]['result'][-1]['content']
        score = RougeEvaluator.rouge_1(ground_truth, prediction)

        recall.append(score.recall)
        precision.append(score.precision)
        f1.append(score.fmeasure)
    
    recall = np.array(recall)
    precision = np.array(precision)
    f1 = np.array(f1)

    return recall.mean(), precision.mean(), f1.mean()