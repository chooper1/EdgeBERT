#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=../glue/
MODEL_TYPE=masked_bert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=MNLI  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI
MOD_NAME=bert

MODEL_NAME=${MOD_NAME}-${MODEL_SIZE}
EPOCHS=10
if [ $MOD_NAME = 'bert' ]
then
  EPOCHS=10
  MODEL_NAME=${MODEL_NAME}-uncased
fi
if [ $MOD_NAME = 'albert' ]
then
  EPOCHS=10
  MODEL_NAME=${MODEL_NAME}-v2
fi

FINAL_THRESH=1

python ../examples/masked_run_glue.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --task_name $DATASET \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size=1 \
  --per_gpu_train_batch_size=8 \
  --learning_rate 2e-5 \
  --num_train_epochs $EPOCHS \
  --overwrite_output_dir \
  --seed 42 \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/two_stage_pruned_${FINAL_THRESH} \
  --save_steps 0 \
  --overwrite_cache \
  --warmup_steps 25000 \
  --mask_scores_learning_rate 1e-2 \
  --initial_threshold 1 --final_threshold ${FINAL_THRESH} \
  --initial_warmup 1 --final_warmup 2 \
  --pruning_method topK --mask_init constant --mask_scale 0.
