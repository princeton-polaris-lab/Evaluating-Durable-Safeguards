from rouge_score import rouge_scorer
import numpy as np
import string
from tqdm import tqdm

from collections import defaultdict

    

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
    category_list = []
    for i in tqdm(range(num_examples)):
        ground_truth = results[i]['ground_truth']
        prediction = results[i]['result'][-1]['content']
        prediction = prediction.replace("Answer", "").strip(string.punctuation).strip()
        score = (1 if prediction == ground_truth else 0)
        category = results[i]['subject']
        category_list.append(category)
        score_list.append(score)
        
    category_scores = defaultdict(float)  # Stores the sum of scores for each category
    category_counts = defaultdict(int)    # Stores the count of examples per category
    for category, score in zip(category_list, score_list):
        category_scores[category] += score
        category_counts[category] += 1
    score_list = np.array(score_list)
    average_scores = {cat: category_scores[cat] / category_counts[cat] for cat in category_scores}
    print("Average Scores per Category:")
    for category, avg_score in average_scores.items():
        print(f"Category {category}: {avg_score:.2f}")
    

    return score_list.mean()*100