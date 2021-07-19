#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=glue/

MODEL_TYPE=albert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=squad  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
EPOCHS=10
if [ $MODEL_TYPE = 'bert' ]
then
  EPOCHS=3
  MODEL_NAME=${MODEL_NAME}-uncased
fi

if [ $MODEL_TYPE = 'albert' ]
then
  EPOCHS=10
  MODEL_NAME=${MODEL_NAME}-v2
fi


python ../examples/masked_run_highway_squad.py \
  --model_type albert_teacher \
  --model_name_or_path $MODEL_NAME \
  --train_file train-v2.0.json \
  --predict_file dev-v2.0.json \
  --data_dir /n/holyscratch01/acc_lab/chooper/albert-py/DeeBERT-albert/scripts/$DATASET/ \
  --do_train \
  --do_eval \
  --do_lower_case \
  --max_seq_length 384 \
  --per_gpu_eval_batch_size 1 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs $EPOCHS \
  --save_steps 0 \
  --seed 42 \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/teacher \
  --overwrite_cache \
  --overwrite_output_dir  
