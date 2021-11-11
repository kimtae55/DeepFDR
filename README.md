# DeepFDR

## Table of Contents
* [Overview](#overview)
* [Installation](#requirements-and-installation)
* [Example](#example)

Folder Structure:

sHMRF- Nearest-neighbor homogeneous HMRFs

cHMRF - High-order-neighborhood non-homogeneous HMRFs

graph - Short-/long-range and graph-based spatial dependencies

DL - W-Net based self-supervised segmentation model

## Overview
## Installation
The project uses ```Python 3.8```
There are some files > 5GB, so please download the code using Dropbox link: [DeepFDR](https://www.dropbox.com/sh/9378gmgy8fb97r9/AABRmGsDHwtiNXH_W55w-igna?dl=0)
Download Dependencies: ```pip install -r requirements.txt```
## Example
Running cHMRF:
1) ```cd``` into the cHMRF directory
2) Generate groundtruth label ```python groundtruth.py rng_seed``` (Specify an integer for the rng_seed, use the same seed for below steps)
3) Generate test statistics 'x' ```python test_statistic.py --seed rng_seed --p_0 0.5 --p_1 0.5 --mu_0 -2.0 --mu_1 2.0 --sig_0 1.0 --sig_1 1.0```
4) Run gem and compute lis ```python gem.py --seed rng_seed```

