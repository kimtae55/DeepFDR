# DeepFDR

The `deepfdr` package provides a fully unsupervised, deep learning based FDR control method designed for spatial multiple testing. We utilize the [W-Net](https://arxiv.org/abs/1711.08506) and the concept of [local index of significance (LIS)](https://academic.oup.com/jrsssb/article/71/2/393/7092902) to connect unsupervised image segmentation to a 3D multiple testing problem that enables false discovery rate (FDR) control with a significant boost in statistical power. We tested our methodology using test statistics generated from FDG-PET data available at [ADNI database](https://adni.loni.usc.edu/). For specific details on methodology or training, please refer to our paper:

T. Kim, H. Shu, Q. Jia, and M. de Leon (2023). [DeepFDR: A Deep Learning-based False Discovery Rate Control Method for Neuroimaging Data]( 	
https://proceedings.mlr.press/v238/kim24b.html). Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, PMLR 238:946-954, 2024.

## Table of Contents
* [Requirements and Installation](#requirements-and-installation)
* [Usage](#usage)
* [Updates](#updates)

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
The following figure illustrates a training session, where the dashed red line indicates the nominal FDR threshold:
![gui_image](https://github.com/kimtae55/DeepFDR/blob/main/figs/gui_sample.png)

## Citing our work using bibtex
@inproceedings{kim2024deepfdr,
  title={DeepFDR: A Deep Learning-based False Discovery Rate Control Method for Neuroimaging Data},
  author={Kim, Taehyo and Shu, Hai and Jia, Qiran and de Leon, Mony},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={946--954},
  year={2024},
  organization={PMLR}
}

## Updates
New updates to improving the model, app stability and user experience will be posted here. 


