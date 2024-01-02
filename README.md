# DeepFDR

The `deepfdr` package provides a fully unsupervised, deep learning based FDR control method designed for spatial multiple testing. We utilize the [W-Net](https://arxiv.org/abs/1711.08506) and the concept of [local index of significance (LIS)](https://academic.oup.com/jrsssb/article/71/2/393/7092902) to connect unsupervised image segmentation to a 3D multiple testing problem that enables false discovery rate (FDR) control with a significant boost in statistical power. We tested our methodology using test statistics generated from FDG-PET data available at [ADNI database](https://adni.loni.usc.edu/). For specific details on methodology or training, please refer to our paper:

T. Kim, H. Shu, Q. Jia, and M. de Leon (2023). [DeepFDR: A Deep Learning-based False Discovery Rate Control Method for Neuroimaging Data](https://arxiv.org/abs/2310.13349v1). arXiv preprint arXiv:2310.13349.

## Table of Contents
* [Requirements and Installation](#requirements-and-installation)
* [Usage](#usage)

## Requirements and Installation
This package was developed using Python 3.9 and Pytorch 1.10.1 - please install the compatible version of Pytorch at [https://pytorch.org/](https://pytorch.org/).
To install the package, please run the following lines:
```bash
# Optional: upgrade pip
pip install --upgrade pip setuptools wheel
```
```bash
git clone https://github.com/kimtae55/DeepFDR
cd $PATH_TO_DeepFDR$
python setup.py install
```

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
![gui_image](https://github.com/kimtae55/DeepFDR/blob/main/figs/gui_example.png)
