# DeepFDR

The Pytorch based `deepfdr` package provides an unsupervised deep learning method for voxel-based multiple testing. We utilize the [W-Net](https://arxiv.org/abs/1711.08506) to connect unsupervised image segmentation to a 3D multiple testing problem that enables false discovery rate (FDR) control with a significant boost in statistical power. We test our methodology using test statistics generated from FDG-PET data available at [ADNI database](https://adni.loni.usc.edu/). 

## Table of Contents
* [Overview](#overview)
* [Folder Structure](#Folder-Structure)
* [Installation](#requirements-and-installation)
* [Example](#example)

## Overview

## Folder Structure
```sHMRF``` contains Nearest-neighbor homogeneous HMRFs, a Python implementation of https://github.com/shu-hai/FDRhmrf

```DeepFDR``` - W-Net based supervised segmentation model

## Installation
The project uses ```Python 3.8```.

There are some files > 5GB, so please download the code using Dropbox link: [DeepFDR](https://www.dropbox.com/sh/9378gmgy8fb97r9/AABRmGsDHwtiNXH_W55w-igna?dl=0)

## SmoothFDR Installation for Linux & HPC users (Singularity with Miniconda)
For any users that use Singularity with Miniconda in an HPC environment, you may find the below steps useful for installing required packages at the right locations:
1) ```git clone https://github.com/tansey/gfl.git```
2) ```git clone https://github.com/tansey/smoothfdr.git```
3) Setup singularity environment https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda
4) Launch: ```singularity exec --overlay taehyo_container.ext3 /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash```
5) Modify makefile in gfl/cpp to ```${CC} -shared -o lib/libgraphfl.so ${OBJ} -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas```
6) Copy the library to a folder, then set its PATH ```export LD_LIBRARY_PATH=/ext3/local/lib:$LD_LIBRARY_PATH```
7) Use ```python setup.py build``` and ```python setup.py install``` to install pygfl and smoothfdr packages

## Example

