#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE=masked_albert
MODEL_SIZE=base
MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
DATASET=MNLI
THRESH="0.8 0.6 0.5 0.4 0.2 0.1"

for FINAL_THRESH in ${THRESH}; do

PATHNAME=./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/${DATASET}/move_pruned_${FINAL_THRESH}
python ../examples/counts_parameters.py \
    --pruning_method  topK \
    --threshold $FINAL_THRESH \
    --serialization_dir $PATHNAME
done
