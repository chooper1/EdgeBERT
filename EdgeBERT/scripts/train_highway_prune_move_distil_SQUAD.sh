#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE=masked_albert  # bert or roberta
MODEL_SIZE=base  # base or large
MOD_NAME=albert

MODEL_NAME=${MOD_NAME}-${MODEL_SIZE}
EPOCHS=10
if [ $MODEL_TYPE = 'bert' ]
then
  EPOCHS=10
  MODEL_NAME=${MODEL_NAME}-uncased
fi
if [ $MOD_NAME = 'albert' ]
then
  EPOCHS=10
  MODEL_NAME=${MODEL_NAME}-v2
fi

SQUAD_DATA=./squad/
DATASET=squad

THRESHOLDS="1"

for FINAL_THRESH in $THRESHOLDS; do
  echo $FINAL_THRESH
  python ../examples/masked_run_squad.py \
    --model_type albert \
    --model_name_or_path $MODEL_NAME \
    --train_file train-v2.0.json \
    --predict_file dev-v2.0.json \
    --data_dir /n/holyscratch01/acc_lab/chooper/albert-py/DeeBERT-albert/scripts/$DATASET/ \
    --do_train \
    --do_eval \
    --do_lower_case \
    --max_seq_length 384 \
    --per_gpu_eval_batch_size=1 \
    --per_gpu_train_batch_size=16 \
    --learning_rate 3e-5 \
    --num_train_epochs $EPOCHS \
    --overwrite_output_dir \
    --seed 42 \
    --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/teacher \
    --save_steps 0 \
    --overwrite_cache \
    --version_2_with_negative 
done
