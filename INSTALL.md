# EdgeBERT Installation Instructions

System requirements (assuming a unix system)
* Conda (tested with version )
* Cuda (tested with version )
* Cudnn (tested with version )

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
