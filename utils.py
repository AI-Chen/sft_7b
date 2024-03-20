# -*- coding: utf-8 -*-
# @Time: 2024/3/20 上午11:07
# @Author: xiaoshuchen
# @email：xschenranker@gmail.com

from transformers.trainer_utils import EvalLoopOutput
import torch

def convert_ids_to_string(token_ids, tokenizer):
    text = ""
    for id in token_ids:
        text += tokenizer._convert_id_to_token(id)
    return text

def print_rank_0(obj):
    if torch.distributed.get_rank() == 0:
        print(obj)

def compute_metrics(eval_output: EvalLoopOutput):
    labels = eval_output.label_ids
    preds = eval_output.predictions
    '''
    print(len(preds))
    print([preds[i] for i in range(len(preds))])
    print([labels[i][0] for i in range(len(preds))])
    '''

    ## This is a recall score for api_call
    correct = []
    for i in range(len(preds)):
        label = labels[i][:list(labels[i]).index(-100) - 1]
        pred = preds[i]
        is_subset = all(elem in pred for elem in label) # check if the correct api_call in pred
        if is_subset:
            correct.append(1)
        else:
            correct.append(0)

    recall = sum(correct) / len(preds)

    metrics = {
        "recall": recall
    }
    return metrics
