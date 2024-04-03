# -*- coding: utf-8 -*-
# @Time: 2024/3/20 上午11:07
# @Author: xiaoshuchen
# @email：xschenranker@gmail.com

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)
import os
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    AutoConfig,
    HfArgumentParser,
    set_seed, )
from transformers.trainer_utils import is_main_process
import sys
import torch
_file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(_file_path, '../'))

from arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from dataloader import ApiDataset, collate_fn
from transformers import AutoTokenizer, BitsAndBytesConfig
#from modeling.falcon.modelling_RW import RWForCausalLM
from modeling.gpt2.modeling_gpt2 import GPT2LMHeadModel as GPT2Model
from modeling.falcon.modeling_falcon import FalconForCausalLM
from modeling.mini_cpm.modeling_minicpm import MiniCPMForCausalLM
from modeling.opt.modeling_opt import OPTForCausalLM
from trainer import Trainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from utils import print_rank_0, compute_metrics

logger = logging.getLogger(__name__)



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.max_seq_len = data_args.max_seq_len

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    if model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        print_rank_0(tokenizer)
    else:
        raise RuntimeError("Config is not initialized correctly.")

    resume_model_path = None
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and any((x.startswith("checkpoint") for x in os.listdir(training_args.output_dir)))
    ):
        if training_args.continue_train:
            ckpts = os.listdir(training_args.output_dir)
            ckpts = list(filter(lambda x: x.startswith("checkpoint"), ckpts))
            ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[1]))
            resume_model_path = os.path.join(training_args.output_dir, ckpts[-1])
        elif not training_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    logger.info(f"Resume model path: {resume_model_path}")

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        #transformers.logging.set_verbosity_error()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)
    if "gpt" in model_args.model_name_or_path:
        model = GPT2Model.from_pretrained(model_args.model_name_or_path).cuda()
    elif "falcon" in model_args.model_name_or_path:
        model = FalconForCausalLM.from_pretrained(model_args.model_name_or_path).cuda()
    elif "mini_cpm" in model_args.model_name_or_path:
        model = FalconForCausalLM.from_pretrained(model_args.model_name_or_path).cuda()
    elif "opt" in model_args.model_name_or_path:
        model = OPTForCausalLM.from_pretrained(model_args.model_name_or_path).cuda()

    train_dataset = ApiDataset(training_args, data_args.data_path, data_args.data_name, mode="train")
    dev_dataset = ApiDataset(training_args, data_args.data_path, data_args.data_name, mode="dev")

    eval_data = {
        'val': dev_dataset,
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_data,
        data_collator=collate_fn(training_args, tokenizer, data_args.max_seq_len, data_args.data_name),
        compute_metrics=compute_metrics,
    )
    # trainer.add_callback(MyTrainerCallback(dataset=train_set, ))
    # Training
    trainer.train()


if __name__ == '__main__':
    main()
