#!/bin/bash

set -e


################################### Environment Setting ###############################
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
# The setting aboved is specific for 4090 GPU
#######################################################################################

####################################### Data INFO #####################################
data_name="huggingface" #choose dataset: huggingface, tensorflow, torchhub
only_api_call=true # true refers to that only output the api_call
target_loss=true # true refers to that only the loss is calculated on the output
quantization=true # true refers to that quantify the parameters in the mode into int4
lora=true # true refers to use lora to train the model
exp_name="test" # experiment name
data_path="/media/xschen/A6503F3E503F1491/xiaoshuchen/DATA/apibench/${data_name}"
echo data_path:$data_path
#######################################################################################

####################################### Model INFO ####################################
#model_type="gpt_base"
#model_name_or_path="/media/xschen/A6503F3E503F1491/xiaoshuchen/MODEL/gpt2_base"
#sep_id=50256
#pad_id=50256
#eod_id=50256

#model_type="gpt_large"
#model_name_or_path="/media/xschen/A6503F3E503F1491/xiaoshuchen/MODEL/gpt2_large"
#sep_id=50256
#pad_id=50256
#eod_id=50256

model_type="falcon"
model_name_or_path="/media/xschen/A6503F3E503F1491/xiaoshuchen/MODEL/falcon_7b"
sep_id=9
pad_id=11
eod_id=11
#######################################################################################

####################################### Basic Params ##################################
#GPU Num per machine
NPROC=2
per_device_train_batch_size=2
per_device_eval_batch_size=1 #32,8

max_seq_len=256
dataloader_num_workers=2

learning_rate=1e-5
logging_steps=20
eval_steps=1000 #20,150
save_steps=10
num_train_epochs=5
gradient_accumulation_steps=2
warmup_steps=1200 #1200

output_dir="ckpt/$exp_name"
if [ ! -d $output_dir ]; then
      mkdir -p $output_dir
fi
echo output_dir: output_dir

RUN="sft_7b"
#######################################################################################

########################################## Run ########################################
if [ $NPROC = 1 ]
then
    distributed_cmd=" "
else
    distributed_cmd=" -m --nproc_per_node $NPROC"
fi

torchrun $distributed_cmd \
  $RUN \
  --data_name $data_name \
  --data_path $data_path \
  --sep_id $sep_id \
  --pad_id $pad_id \
  --eod_id $eod_id \
  --model_type $model_type \
  --model_name_or_path $model_name_or_path \
  --output_dir $output_dir \
  --eval_accumulation_steps 1 \
  --logging_steps $logging_steps \
  --save_total_limit 1 \
  --evaluation_strategy "steps" \
  --eval_steps $eval_steps \
  --save_steps $save_steps \
  --num_train_epochs $num_train_epochs \
  --learning_rate $learning_rate \
  --lr_scheduler_type "cosine_with_restarts" \
  --warmup_steps $warmup_steps \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --max_seq_len $max_seq_len \
  --per_device_train_batch_size $per_device_train_batch_size \
  --per_device_eval_batch_size $per_device_eval_batch_size \
  --overwrite_output_dir \
  --dataloader_num_workers $dataloader_num_workers \
  --fp16 \
  --only_api_call $only_api_call \
  --target_loss $target_loss \
  --quantization $quantization \
  --lora $lora
#######################################################################################
