# EdgeBERT

This repository contains the software code used to train and evaluate models in the paper:

**EdgeBERT: Sentence-Level Energy Optimizations for Latency-Aware Multi-Task NLP Inference** (https://arxiv.org/abs/2011.14203).

# Installation Instructions:
See INSTALL.md for detailed installation instructions.

# Producing a lookup table for entropy prediction
To produce a lookup table for entropy prediction:

1. Open entropypredictor.ipynb in Google Colab and load the desired training and test entropy datasets.

2. Run all cells in this notebook and download the resulting csv file.

# Training and evaluating models (example is with SST-2)

Change into the EdgeBERT/scripts directory and follow the steps in the README.md file.


1. Run download_glue.sh to clone https://github.com/nyu-mll/GLUE-baselines.git and download the required datasets.

2. Run train_teacher_sst2.sh to train a teacher model.

3. Run train_sst2.sh to train the full model.

4. Run bertarize_sst2.sh to prune the full model.

5. Run eval_sst2_ee.sh to evaluate the full model with early exit.

6. Load the lookup table (csv file) produced by the entropypredictor.ipynb script.

7. Run eval_sst2_ep.sh to evaluate the full model with entropy prediction.

8. (Optional) Run eval_sst2_ep_predlayer.sh to identify the average predicted exit layer used by DVFS.
