#!/bin/bash

conda create -n unihr -y python=3.8
conda activate unihr

conda install scikit-learn -y
conda install numpy -y
conda install pandas -y
conda install tqdm -y

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#pip install torch-1.10.0+cu113-cp38-cp38-linux_x86_64.whl
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install sympy
pip install psutil
pip install networkx
