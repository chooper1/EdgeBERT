#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE=masked_albert  
MODEL_SIZE=base
MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
DATASET=MNLI
FINAL_THRESH=0.8

PATHNAME=./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/two_stage_pruned_adaptive_one_class_true_${FINAL_THRESH}

python ../examples/bertarize.py \
    --pruning_method topK \
    --threshold $FINAL_THRESH \
    --model_name_or_path $PATHNAME \
