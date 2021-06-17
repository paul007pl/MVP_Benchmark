#!/bin/bash

conda create -n mvp python=3.7
conda activate mvp
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

