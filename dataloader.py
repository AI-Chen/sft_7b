# -*- coding: utf-8 -*-
# @Time: 2024/3/20 上午11:10
# @Author: xiaoshuchen
# @email：xschenranker@gmail.com

import os
import json
import random
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from utils import print_rank_0, convert_ids_to_string

class ApiDataset(Dataset):
    def __init__(self, args, data_path, data_name, mode="train"):
        self.args = args
        self.mode = mode
        self.data_name = data_name
        if mode == "train":
            self.samples = self._create_examples(data_path + "_{}.json".format(mode))
        elif mode == "dev":
            self.samples = self._create_examples(data_path + "_{}.json".format("eval"))
    def _create_examples(self, filename):
        """Creates examples for the training or dev sets."""
        examples = []
        fin = open(filename, "r")
        for (i, line) in enumerate(fin):
            line = json.loads(line.strip())
            sample = line['code']
            api_call = line["api_call"]
            examples.append([sample, api_call])
        fin.close()
        if self.mode == "train":
            random.shuffle(examples)
        return examples

    def _get_samples(self):
        return self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, api_call = self.samples[idx]
        return {
            "sample": sample,
            "api_call": api_call,
            "mode": self.mode
        }


@dataclass
class collate_fn():
    def __init__(self, args, tokenizer, max_seq_length, data_name):
        self.args = args
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_id = args.pad_id
        self.eod_id = args.eod_id
        self.sep_id = args.sep_id

    def encode_text(self, text, pad_id=None):
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        token_ids = token_ids[:self.max_seq_length -1] + [self.eod_id]
        attention_mask = [1] * len(token_ids)
        pad_length = self.max_seq_length - len(token_ids)
        if pad_id is None:
            token_ids = token_ids + [self.pad_id] * pad_length
        else:
            token_ids = token_ids + [-100] * pad_length
        attention_mask = attention_mask + [0] * pad_length
        return token_ids, attention_mask

    def process_text_train(self, example):
        sample = example["sample"]
        return self.encode_text(sample)

    def process_text_dev(self, example):
        sample, api_call = example["sample"], example["api_call"]
        sample_ids, sample_attn_mask = self.encode_text(sample)
        instruct = sample.split("###Output: ")[0] + "###Output: "
        ins_ids, ins_attn_mask = self.encode_text(instruct)
        api_ids, _ = self.encode_text(api_call, pad_id=-100)

        return sample_ids, sample_attn_mask, ins_ids, ins_attn_mask, api_ids

    def __call__(self, features):
        mode = features[0]["mode"]
        if mode == "train":
            fea_list = [[] for _ in range(2)]
            for fea in features:
                encode_feas = self.process_text_train(fea)
                for idx, encode_fea in enumerate(encode_feas):
                    fea_list[idx].append(encode_fea)
            batch_data = {
                    'input_ids': torch.tensor(fea_list[0]),
                    'attention_mask': torch.tensor(fea_list[1]),
                    'labels': torch.tensor(fea_list[0]),  # Note that the labels **are shifted** inside the model, i.e. you can set `labels = input_ids`
            }
        elif mode == "dev":
            fea_list = [[] for _ in range(5)]
            for fea in features:
                encode_feas = self.process_text_dev(fea)
                for idx, encode_fea in enumerate(encode_feas):
                    fea_list[idx].append(encode_fea)
            batch_data = {
                'sample_ids': torch.tensor(fea_list[0]),
                'sample_attn_mask': torch.tensor(fea_list[1]),
                'sample_labels': torch.tensor(fea_list[0]),
                'ins_ids': torch.tensor(fea_list[2]),
                'ins_attn_mask': torch.tensor(fea_list[3]),
                'api_ids': torch.tensor(fea_list[4])
            }
        return batch_data