# PCLUDA: A Pseudo-Label Consistency Learning- Based Unsupervised Domain Adaptation Method for Cross-Domain Optical Remote Sensing Image Retrieval
This repository is the official implementation of [PCLUDA: A Pseudo-Label Consistency Learning-Based Unsupervised Domain Adaptation Method for Cross-Domain Optical Remote Sensing Image Retrieval](https://ieeexplore.ieee.org/document/10003219) (IEEE TGRS 2023).


<b>Authors</b>: Dongyang Hou, Siyuan Wang, Xueqing Tian and Huaqiao Xing


## Requirements
- This code is written for `python3`.
- pytorch >= 1.7.0
- torchvision
- numpy, prettytable, tqdm, scikit-learn, matplotlib, argparse, h5py


## Data Preparing
Download dataset from the following link (code is chk8):

[BaiduYun](https://pan.baidu.com/s/1YbsJZQEFaLyl3HRE3uBsbQ)

## Training and Evaluating
The pipeline for training with PCLUDA is the following:

1. Train the model. For example, to run an experiment for UCM_LandUse dataset (source domain) and AID dataset (target domain),  run:

  `python pcluda.py /your_path/PCLUDA_dataset/ -s UCMD -t AID -a resnet50 --epochs 30 --seed 1 --log logs/pcluda/ucmd_aid`

2. Evaluate the classification performance of the model. 

  `python pcluda.py /your_path/PCLUDA_dataset/ -s UCMD -t AID -a resnet50 --epochs 30 --seed 1 --log logs/pcluda/ucmd_aid --phase test`

3. Image Retrieval Test. The retrieval code is under continuous optimization.

  `python pcluda.py /your_path/PCLUDA_dataset/ -s UCMD -t AID -a resnet50 --epochs 30 --seed 1 --log logs/pcluda/ucmd_aid --phase retrieval`


## Acknowledgment
This code is heavily borrowed from [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)

## Citation
If you find our work useful in your research, please consider citing our paper:

```
@article{hou2023pcluda,
  title={PCLUDA: A Pseudo-label Consistency Learning-Based Unsupervised Domain Adaptation Method for Cross-domain Optical Remote Sensing Image Retrieval},
  author={Hou, Dongyang and Wang, Siyuan and Tian, Xueqing and Xing, Huaqiao},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```
## Contact
Please contact houdongyang1986@163.com if you have any question on the codes.
