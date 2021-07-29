# EdgeBERT Installation Instructions

System requirements (assuming a unix system)
* Anaconda3 (tested with version 5.0.1)
* Cuda (tested with version 10.0.130)
* Cudnn (tested with version 7.4.1.5)

Run the following commands:

```
conda create --name test-edge python=3.7
source activate test-edge

pip install torch
pip install tensorboard
pip install -U scikit-learn

cd EdgeBERT/transformers
python setup.py install
```
