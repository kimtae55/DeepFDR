# DeepFDR

## Table of Contents
* [Overview](#overview)
* [Folder Structure](#Folder-Structure)
* [Installation](#requirements-and-installation)
* [Example](#example)

## Overview

## Folder Structure
```sHMRF``` contains Nearest-neighbor homogeneous HMRFs, a Python implementation of https://github.com/shu-hai/FDRhmrf

```cHMRF``` - High-order-neighborhood non-homogeneous HMRFs

```graph``` - Short-/long-range and graph-based spatial dependencies

```DL``` - W-Net based supervised segmentation model

```cosmos``` - W-Net based multi-objective optimization (MOO) model 

```smoothfdr``` - Local smoothing multiple testing method developed by https://github.com/tansey/smoothfdr

## Installation
The project uses ```Python 3.8```.

There are some files > 5GB, so please download the code using Dropbox link: [DeepFDR](https://www.dropbox.com/sh/9378gmgy8fb97r9/AABRmGsDHwtiNXH_W55w-igna?dl=0)

## SmoothFDR Installation for Linux & HPC users(Singularity with Miniconda)
1) ```git clone https://github.com/tansey/gfl.git```
2) ```git clone https://github.com/tansey/smoothfdr.git```
3) Setup singularity environment https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda
4) Launch: ```singularity exec --overlay taehyo_container.ext3 /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash```
5) Modify makefile in gfl/cpp to ```${CC} -shared -o lib/libgraphfl.so ${OBJ} -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas```
6) Copy the library to a folder, then set its PATH ```export LD_LIBRARY_PATH=/ext3/local/lib:$LD_LIBRARY_PATH```
7) Use ```python setup.py build``` and ```python setup.py install``` to install pygfl and smoothfdr packages

Download Dependencies: ```pip install -r requirements.txt```
## Example
Running cHMRF:
1) ```cd``` into the cHMRF directory
2) Generate groundtruth label ```python groundtruth.py rng_seed``` (Specify an integer for the rng_seed, use the same seed for below steps)
3) Generate test statistics 'x' ```python test_statistic.py --seed rng_seed --p_0 0.5 --p_1 0.5 --mu_0 -2.0 --mu_1 2.0 --sig_0 1.0 --sig_1 1.0```
4) Run gem and compute lis ```python gem.py --seed rng_seed```
