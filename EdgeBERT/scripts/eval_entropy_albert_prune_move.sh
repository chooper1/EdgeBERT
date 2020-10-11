#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=../glue/
MODEL_TYPE=masked_albert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=MNLI  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI
MOD_NAME=albert

MODEL_NAME=${MOD_NAME}-${MODEL_SIZE}
EPOCHS=10
if [ $MODEL_TYPE = 'bert' ]
then
  EPOCHS=3
  MODEL_NAME=${MODEL_NAME}-uncased
fi
if [ $MOD_NAME = 'albert' ]
then
  EPOCHS=10
  MODEL_NAME=${MODEL_NAME}-v2
fi

#ENTROPIES="0 0.1 0.2 0.5"
#THRESHOLDS="1 0.50 0.25 0.10 0.05 0.03"

ENTROPIES="0"
THRESHOLDS="1"

for FINAL_THRESH in $THRESHOLDS; do
  for ENTROPY in $ENTROPIES; do
    echo $FINAL_THRESH
    echo $ENTROPY
    python ../examples/masked_run_highway_glue.py \
      --model_type albert \
      --model_name_or_path ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/bertarized_two_stage_pruned_${FINAL_THRESH} \
      --task_name $DATASET \
      --do_eval \
      --do_lower_case \
      --data_dir $PATH_TO_DATA/$DATASET \
      --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/bertarized_two_stage_pruned_${FINAL_THRESH} \
      --plot_data_dir ./plotting/ \
      --max_seq_length 128 \
      --per_gpu_eval_batch_size=1 \
      --early_exit_entropy $ENTROPY \
      --eval_highway \
      --overwrite_cache \
      --per_gpu_eval_batch_size=1
  done
done
