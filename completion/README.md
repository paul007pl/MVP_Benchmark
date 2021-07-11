# *MVP Benchmark:* Point Cloud Completioin
<p align="center"> 
<img src="images/mvp.png">
</p>


We include the following methods for point cloud completion:

[1] [PCN](https://github.com/wentaoyuan/pcn);&nbsp;&nbsp; [2] [ECG](https://github.com/paul007pl/ECG);&nbsp;&nbsp; [3] [VRCNet](https://github.com/paul007pl/VRCNet)


### MVP Completion Dataset
<!-- Download the MVP completion dataset by the following commands:
```
cd data; sh download_data.sh
``` -->
Download the MVP completion dataset [Google Drive](https://drive.google.com/drive/folders/1XxZ4M_dOB3_OG1J6PnpNvrGTie5X9Vk_) or [百度网盘](https://pan.baidu.com/s/18pli79KSGGsWQ8FPiSW9qg)&nbsp;&nbsp;(code: p364) to the folder "data".

The data structure will be:
```
data
├── MVP_Train_CP.h5
|    ├── incomplete_pcds (62400, 2048, 3)
|    ├── complete_pcds (2400, 2048, 3)
|    └── labels (62400,)
├── MVP_Test_CP.h5
|    ├── incomplete_pcds (41600, 2048, 3)
|    ├── complete_pcds (1600, 2048, 3)
|    └── labels (41600,)
└── MVP_ExtraTest_Shuffled_CP.h5
     ├── incomplete_pcds (59800, 2048, 3)
     └── labels (59800,)
```

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| category | airplane | cabinet | car | chair | lamp | sofa | table | watercraft | bed | bench | bookshelf | bus | guitar | motorbike | pistol | skateboard | 
| \#train  | 5200 | 5200 | 5200 | 5200 | 5200 | 5200 | 5200 | 5200 | 2600 | 2600 | 2600 | 2600 | 2600 | 2600 | 2600 | 2600 |
| \#test  | 3900 | 3900 | 3900 | 3900 | 3900 | 3900 | 3900 | 3900 | 1300 | 1300 | 1300 | 1300 | 1300 | 1300 | 1300 | 1300 |



<!-- **Partial point clouds** & **Complete point clouds**

<center class="half">
  <figure>
    <img src="images/partial_pcds.gif", width=400><img src="images/complete_pcds.gif", width=400>
  </figure>
</center> -->

<!-- Partial point clouds | Complete point clouds
:-------------------------:|:---------------- ---------:
![](./images/partial_pcds.gif) | ![](./images/complete_pcds.gif) -->


### Usage
+ To train a model: run `python train.py -c ./cfgs/*.yaml`, e.g. `python train.py -c ./cfgs/pcn.yaml`
+ To test a model: run `python test.py -c ./cfgs/*.yaml`, e.g. `python test.py -c ./cfgs/pcn.yaml`
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
[1] [PCN](https://github.com/wentaoyuan/pcn)    
[2] [ECG](https://github.com/paul007pl/ECG)    
[3] [VRCNet](https://github.com/paul007pl/VRCNet)   

