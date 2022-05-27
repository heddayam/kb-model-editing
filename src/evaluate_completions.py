from rouge_score import rouge_scorer
import pandas as pd 
import numpy as np
import json
import pdb

kelm_gold = "../data/formatted/kelm_test_set.jsonl"
kelm_baseline = "../data/formatted/kelm_test_completions.jsonl"
kelm_rome = "../data/formatted/kelm_edited_test_completions.jsonl"

# Construct dataset with the prompt, gold_completion, baseline_completion, rome_generated_completion,
def construct_completions_df(kelm_gold, kelm_baseline, kelm_rome):
    with open(kelm_gold, 'r') as gold_file, open(kelm_baseline, 'r') as baseline_file, open(kelm_rome, 'r') as rome_file:
        gold_list = list(gold_file)
        baseline_list = list(baseline_file)
        rome_list = list(rome_file)

    completions = pd.DataFrame()
    prompts = []
    gold_completions = []
    baseline_completions = []
    rome_completions = []

    for json_gold, json_baseline, json_rome in zip(gold_list, baseline_list, rome_list):
        gold_instance = json.loads(json_gold)
        baseline_instance = json.loads(json_baseline)
        rome_instance = json.loads(json_rome)
        
        prompts.append(baseline_instance['prompt'])
        gold_completions.append(gold_instance['target_new']['str'])
        baseline_completions.append(baseline_instance['completion'])
        rome_completions.append(rome_instance['rome_completions'])
        
    completions = pd.DataFrame.from_dict(
        {
            'prompt': prompts, 
            "gold_completions": gold_completions, 
            "baseline_completions": baseline_completions, 
            "rome_completions": rome_completions
        }
    )

    return completions



def evaluate(df):
    '''Function to evaluate generated completionusing ROUGE-1 metric'''
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    bl_precision = []
    bl_recall = []
    bl_fmeasure = []
    rome_precision = []
    rome_recall = []
    rome_fmeasure = []
    for _, line in df.iterrows():
        rouge1_bl_scores = scorer.score(line.gold_completions, line.baseline_completions)['rouge1']
        rouge1_rome_scores = scorer.score(line.gold_completions, line.rome_completions)['rouge1']
        bl_precision.append(rouge1_bl_scores.precision)
        bl_recall.append(rouge1_bl_scores.recall)
        bl_fmeasure.append(rouge1_bl_scores.fmeasure)
        rome_precision.append(rouge1_rome_scores.precision)
        rome_recall.append(rouge1_rome_scores.recall)
        rome_fmeasure.append(rouge1_rome_scores.fmeasure)

    rouge1_df = pd.DataFrame.from_dict({
            "bl_precision": bl_precision,
            "bl_recall": bl_recall,
            "bl_fmeasure": bl_fmeasure,
            "rome_precision": rome_precision,
            "rome_recall": rome_recall,
            "rome_fmeasure": rome_fmeasure
        })
    return rouge1_df

completions = construct_completions_df(kelm_gold, kelm_baseline, kelm_rome)
df = evaluate(completions)
print(df.mean())
