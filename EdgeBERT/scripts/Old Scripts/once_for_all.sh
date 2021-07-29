#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE=masked_albert  
MODEL_SIZE=base
MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
DATASET=MNLI
FINAL_THRESH="400 600"

for THRESH in $FINAL_THRESH; do

PATHNAME=./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/soft_move_pruned_${THRESH}_distil

python ../examples/bertarize.py \
    --pruning_method sigmoied_threshold \
    --threshold 0.1 \
    --model_name_or_path $PATHNAME 
done
