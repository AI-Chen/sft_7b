# -*- coding: utf-8 -*-
# @Time: 2024/3/20 上午11:51
# @Author: xiaoshuchen
# @email：xschenranker@gmail.com

import os
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from transformers.trainer import Trainer
from utils import  print_rank_0, convert_ids_to_string

import logging
import time
logger = logging.getLogger(__name__)

class Trainer(Trainer):
    def compute_loss(self, model, inputs):
        model_out = model(**inputs)
        loss = model_out.loss
        return loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            '''
            if is_torch_tpu_available():
                xm.mark_step()
            '''
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = {}
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    local_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    # logger.info(local_metrics)
                    metrics.update(local_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            logger.info(metrics)
            self.log(metrics)
            self._report_to_hp_search(trial, epoch, metrics)  # self.state.global_step

        ''' 
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=logs)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
         '''
    def prediction_step(
            self,
            model,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ):
        assert prediction_loss_only == False
        assert ignore_keys is None

        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("api_ids")

        input_data = {
            "input_ids": inputs.get("sample_ids"),
            "attention_mask": inputs.get("sample_attn_mask"),
            "labels": inputs.get("sample_labels")
        }
        model_out = model(**input_data)
        loss = model_out.loss.detach()

        instruction = inputs.get("ins_ids")

        #loss = None
        preds = []
        sequence_lengths = (torch.ne(instruction, self.args.eod_id).sum(-1)).to(labels.device)
        for bidx, bdata in enumerate(instruction):
            seq_l_idx = sequence_lengths[bidx]
            # print(convert_ids_to_string(bdata[:seq_l_idx]))
            # print(a)
            output = model.generate(input_ids=bdata[:seq_l_idx].unsqueeze(0), max_length=self.args.max_seq_len).squeeze()
            output = output.tolist() + (self.args.max_seq_len - len(output)) * [self.args.pad_id]
            preds.append(output)
        preds = torch.tensor(preds).cuda(device=labels.device)
        return (loss, preds, labels)