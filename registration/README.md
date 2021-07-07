# Partial-to-Partial Point Cloud Registration
<p align="center"> 
<img src="images/registration.png">
</p>

We include the following methods for point cloud registration:

[1] [DCP](https://github.com/WangYueFt/dcp); [2] [DeepGMR](https://github.com/wentaoyuan/deepgmr); [3] [IDAM](https://github.com/jiahaowork/idam)


### MVP Registration Dataset
Download the MVP registration dataset by the following commands:
```
cd data; sh download_data.sh
```


### Usage
+ To train a model: run `python train.py -c *.yaml`, e.g. `python train.py -c pcn.yaml`
+ To test a model: run `python test.py -c *.yaml`, e.g. `python test.py -c pcn.yaml`
+ Config for each algorithm can be found in `cfgs/`.
+ `run_train.sh` and `run_test.sh` are provided for SLURM users. 


## Citation
If you find our code useful, please cite our paper:
```bibtex
@article{pan2021variational,
  title={Variational Relational Point Completion Network},
  author={Pan, Liang and Chen, Xinyi and Cai, Zhongang and Zhang, Junzhe and Zhao, Haiyu and Yi, Shuai and Liu, Ziwei},
  journal={arXiv preprint arXiv:2104.10154},
  year={2021}
}
```

## License
Our code is released under Apache-2.0 License.


## Acknowledgement
We include the following algorithms:  
[1] [DCP](https://github.com/WangYueFt/dcp)     
[2] [DeepGMR](https://github.com/wentaoyuan/deepgmr)     
[3] [IDAM](https://github.com/jiahaowork/idam)    
