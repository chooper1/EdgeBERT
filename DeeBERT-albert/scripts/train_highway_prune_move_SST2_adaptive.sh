#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=/n/holyscratch01/acc_lab/ttambe/models/DeeBERT-albert/glue/
MODEL_TYPE=masked_albert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=SST-2  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI
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

THRESHOLDS="1 0.8 0.6 0.4 0.2 0.1"

for FINAL_THRESH in $THRESHOLDS; do
  echo $FINAL_THRESH
  python ../examples/masked_run_highway_glue.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME \
    --task_name $DATASET \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $PATH_TO_DATA/$DATASET \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=1 \
    --per_gpu_train_batch_size=32 \
    --learning_rate 2e-5 \
    --num_train_epochs $EPOCHS \
    --overwrite_output_dir \
    --seed 42 \
    --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/two_stage_pruned_adaptive_${FINAL_THRESH} \
    --plot_data_dir ./plotting/ \
    --save_steps 0 \
    --eval_after_first_stage \
    --warmup_steps 1000 \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold ${FINAL_THRESH} \
    --initial_warmup 1 --final_warmup 2 \
    --pruning_method topK --mask_init constant --mask_scale 0. \
    --adaptive_span_ramp 256 \
    --max_span 512 \
    --adaptive
    #--overwrite_cache \
done
