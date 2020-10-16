#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE=masked_albert
MODEL_SIZE=base
MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
DATASET=MNLI
THRESH="1 10"

for FINAL_THRESH in ${THRESH}; do
PATHNAME=./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/${DATASET}/soft_move_pruned_${FINAL_THRESH}
python ../examples/count_parameters.py \
    --pruning_method sigmoied_threshold \
    --threshold 0.1 \
    --model_name_or_path $PATHNAME \
done
