# EdgeBERT

This repository contains the software code used to train and evaluate models in the paper:

**EdgeBERT: Sentence-Level Energy Optimizations for Latency-Aware Multi-Task NLP Inference** (https://arxiv.org/abs/2011.14203).

# Installation Instructions:
See INSTALL.md for detailed installation instructions.

# Producing a lookup table for entropy prediction
To produce a lookup table for entropy prediction:

1. Open Entropy_LUT/entropypredictor.ipynb in Google Colab and load the desired training and test entropy datasets.

2. Run all cells in this notebook and download the resulting csv file.

# Training and evaluating models (example is with SST-2)

Change into the EdgeBERT/scripts directory and follow the steps in the README.md file.
