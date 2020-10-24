#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE=masked_albert
MODEL_SIZE=base
DATASET=MNLI
THRESH="1 10" #note that final_thresh here is the lambda value

for FINAL_THRESH in $THRESH;
do

PATHNAME=./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/soft_move_pruned_${FINAL_THRESH}

python ../examples/bertarize.py \
    --pruning_method sigmoied_threshold \
    --threshold 0.1  \
    --model_name_or_path $PATHNAME \

done
