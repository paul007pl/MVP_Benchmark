#!/bin/bash

conda create -n mvp python=3.7
conda activate mvp
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

# setup completion
cd completion
pip install -r requirements.txt

# cd ./utils/metrics/CD/chamfer3D/
# # SLURM users
# sh run_build.sh
# # or 
# # python setup.py install
# cd ../../../../

# cd ./utils/metrics/EMD/
# # SLURM users
# sh run_build.sh
# # or 
# # python setup.py install
# cd ../../../

cd ./utils/mm3d_pn2/
sh setup.sh

# SLURM users
sh run_build.sh
# or 
# python setup.py develop

cd ../../

# setup registration


