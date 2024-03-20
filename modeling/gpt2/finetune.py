# -*- coding: utf-8 -*-
'''
@Time    : 2023/6/6 21:42
@Author  : xiaoshuchen@tencent.com
@File    : pretrain.py

'''
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_name: str = field(
        default="CommonsenseQA", metadata={"help": "The name of training and evaluation sets"}        
    )
    data_path: str = field(
        default="/apdcephfs_nj2/share_300595189/xiaoshuchen/DATA/gpt_test/CommonsenseQA", metadata={"help": "The path of training and evaluation data"}
    )
    data_size: Optional[str] = field(
        default="1",
        metadata={"help": "the percent of dataset used by model"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_len: Optional[int] = field(
        default=2048,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    max_answer_len: Optional[int] = field(
        default=150,
        metadata={"help": "the max length of answer (label + rationale)"}
    )


@dataclass
class ModelArguments:
    model_type: Optional[str] = field(
                    default=None, metadata={"help": "The finetune model type"}
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

@dataclass
class PreTrainingArguments(TrainingArguments):
    cache_chunk_size: int = field(default=-1)
    gradient_checkpointing: bool = field(default=False)
    continue_train: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)  # important
    ddp_find_unused_parameters: bool = field(default=True)
    seed: Optional[int] = field(default=1234)
    data_argument: Optional[bool] = field(default=False)
    reason: Optional[bool] = field(default=True)
    cot: Optional[bool] = field(default=True)
    distill_type: Optional[str] = field(default="lm")
    alpha: Optional[str] = field(default="0.9")
    cls_num: Optional[int] = field(default=4)
    eod_id: Optional[int] = field(default=2)
    sep_id: Optional[int] = field(default=0)
    pad_id: Optional[int] = field(default=1)
