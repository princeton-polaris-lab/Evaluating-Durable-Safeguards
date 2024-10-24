from rouge_score import rouge_scorer
import numpy as np
import string
from tqdm import tqdm


    

# def qa_acc_metric(results):
    
#     num_examples = len(results)
#     score_list = []

#     for i in tqdm(range(num_examples)):
#         ground_truth = results[i]['ground_truth']
#         prediction = results[i]['result'][-1]['content']
#         prediction = prediction.split("\n")
#         selected_prediction = [s for s in prediction if "Answer" in s]
#         if len(selected_prediction) == 0:
#             score = 0
#         else:
#             prediction = selected_prediction[0]
#             prediction = prediction.replace("Answer", "").strip(string.punctuation).strip()
#             score = (1 if prediction == ground_truth else 0)

#         score_list.append(score)
    
#     score_list = np.array(score_list)
    

#     return score_list.mean()*100



def qa_acc_metric(results):
    
    num_examples = len(results)
    score_list = []

    for i in tqdm(range(num_examples)):
        ground_truth = results[i]['ground_truth']
        prediction = results[i]['result'][-1]['content']
        prediction = prediction.replace("Answer", "").strip(string.punctuation).strip()
        score = (1 if prediction == ground_truth else 0)

        score_list.append(score)
    
    score_list = np.array(score_list)
    

    return score_list.mean()*100