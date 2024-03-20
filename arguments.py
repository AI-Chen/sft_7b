# -*- coding: utf-8 -*-
# @Time: 2024/3/20 上午11:09
# @Author: xiaoshuchen
# @email：xschenranker@gmail.com

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
        default="./", metadata={"help": "The path of training and evaluation data"}
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
class TrainingArguments(TrainingArguments):
    cache_chunk_size: int = field(default=-1)
    gradient_checkpointing: bool = field(default=False, metadata={"help": "if gradient_checkpointing or not"})
    continue_train: bool = field(default=False, metadata={"help": "if continue_train or not"})
    remove_unused_columns: bool = field(default=False)  # important
    ddp_find_unused_parameters: bool = field(default=True)
    seed: Optional[int] = field(default=1234, metadata={"help": "ramdom seed"})
    eod_id: Optional[int] = field(default=2, metadata={"help": "The end of text id of the model"})
    sep_id: Optional[int] = field(default=0, metadata={"help": "The id for separating the text into instruction and answer"})
    pad_id: Optional[int] = field(default=1, metadata={"help": "The pading id of the model"})