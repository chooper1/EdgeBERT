# EdgeBERT Installation Instructions

System requirements (assuming a unix system)
* Conda (tested with )
* Cuda (tested with )
* Cudnn (tested with )

Run the following commands:
`conda create --name test-edge python=3.7
source activate test-edge
pip install torch
pip install tensorboard
pip install -U scikit-learn
cd EdgeBERT/transformers
python setup.py install
`
