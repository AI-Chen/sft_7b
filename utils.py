# -*- coding: utf-8 -*-
# @Time: 2024/3/20 上午11:07
# @Author: xiaoshuchen
# @email：xschenranker@gmail.com

from transformers.trainer_utils import EvalLoopOutput
import torch
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from transformers import HfArgumentParser

import jsonlines as jsonl

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

def convert_ids_to_string(token_ids):
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text

def print_rank_0(obj):
    if torch.distributed.get_rank() == 0:
        print(obj)



def compute_metrics(eval_output: EvalLoopOutput):
    fout = jsonl.open(training_args.output_dir + "/pred_result.jsonl", mode="w")
    labels = eval_output.label_ids
    preds = eval_output.predictions

    ## This is a recall score for api_call
    correct = []
    for i in range(len(preds)):
        label = labels[i][:list(labels[i]).index(-100) - 1]
        pred = preds[i]
        is_subset = all(elem in pred for elem in label) # check if the correct api_call in pred
        fout.write({"label": convert_ids_to_string(label), "pred":convert_ids_to_string(pred), "is_subset": is_subset})
        if is_subset:
            correct.append(1)
        else:
            correct.append(0)
    fout.close()
    recall = sum(correct) / len(correct)

    metrics = {
        "recall": recall
    }
    return metrics
