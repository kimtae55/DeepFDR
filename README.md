# DeepFDR

The Pytorch based `deepfdr` package provides a fully unsupervised deep learning method for voxel-based multiple testing. We utilize the [W-Net](https://arxiv.org/abs/1711.08506) and the concept of [local index of significance (LIS)](https://academic.oup.com/jrsssb/article/71/2/393/7092902) to connect unsupervised image segmentation to a 3D multiple testing problem that enables false discovery rate (FDR) control with a significant boost in statistical power. We test our methodology using test statistics generated from FDG-PET data available at [ADNI database](https://adni.loni.usc.edu/). For specific details on methodology or training, please refer to our [paper](https://arxiv.org/abs/2310.13349v1) on arxiv. 

## Table of Contents
* [Installation](#requirements-and-installation)
* [Usage](#usage)

## Installation
This package was developed using Python 3.9 and Pytorch 1.10.1 - please install the compatible version of Pytorch at [https://pytorch.org/](https://pytorch.org/)
To install the package, please run the following lines:
```bash
git clone [https://github.com/rmarkello/snfpy.git](https://github.com/kimtae55/DeepFDR)
cd $PATH_TO_DeepFDR$
python setup.py install
```

## SmoothFDR Installation for Linux & HPC users (Singularity with Miniconda)
For any users that use Singularity with Miniconda in an HPC environment, you may find the below steps useful for installing required packages at the right locations:
1) ```git clone https://github.com/tansey/gfl.git```
2) ```git clone https://github.com/tansey/smoothfdr.git```
3) Setup singularity environment https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda
4) Launch: ```singularity exec --overlay taehyo_container.ext3 /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash```
5) Modify makefile in gfl/cpp to ```${CC} -shared -o lib/libgraphfl.so ${OBJ} -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas```
6) Copy the library to a folder, then set its PATH ```export LD_LIBRARY_PATH=/ext3/local/lib:$LD_LIBRARY_PATH```
7) Use ```python setup.py build``` and ```python setup.py install``` to install pygfl and smoothfdr packages

## Usage
The software offers two distinct training modalities: an interactive training interface leveraging Dash and Plotly for a dynamic, web-app-based training, and a standalone training option for those preferring operation without the web application. If one wants to activate the GUI, use the ```train_gui``` module:
```bash
python train_gui.py --labelpath {optional groundtruth file if using for simulation}
                    --datapath {input test statistics file}
                    --savepath {directory path for saving results}
```
If one wants to train without visualization, use the ```train``` module:
```bash
python train.py --labelpath {optional groundtruth file if using for simulation}
                --datapath {input test statistics file}
                --savepath {directory path for saving results}
```


