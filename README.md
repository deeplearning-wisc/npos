# Non-parametric Outlier Synthesis (To be updated)

This codebase provides a Pytorch implementation for the paper NPOS: [Non-parametric Outlier Synthesis](https://openreview.net/forum?id=JHklpEZqduQ) at ICLR 2023.

### Abstract

Out-of-distribution (OOD) detection is indispensable for safely deploying machine learning models in the wild. One of the key challenges is that models lack supervision signals from unknown data, and as a result, can produce overconfident predictions on OOD data. Recent work on outlier synthesis modeled the feature space as parametric Gaussian distribution, a strong and restrictive assumption that might not hold in reality. In this paper, we propose a novel framework, non-parametric outlier synthesis (NPOS), which generates artificial OOD training data and facilitates learning a reliable decision boundary between ID and OOD data. Importantly, our proposed synthesis approach does not make any distributional assumption on the ID embeddings, thereby offering strong flexibility and generality. We show that our synthesis approach can be mathematically interpreted as a rejection sampling framework. Extensive experiments show that NPOS can achieve superior OOD detection performance, outperforming the competitive rivals by a significant margin. 

# Setup

## Required Packages

Our experiments are conducted on Ubuntu Linux 20.04 with Python 3.8 and Pytorch 1.11. Besides, the following packages are required to be installed:

- [clip](https://github.com/openai/CLIP)
- [faiss](https://github.com/facebookresearch/faiss)

## Quick Start

Remarks: This is the initial version of our codebase, and while the scripts are functional, there is much room for improvement in terms of streamlining the pipelines for better efficiency. Additionally, there are unused parts that we plan to remove in an upcoming cleanup soon. Stay tuned for more updates.

### Data Preparation

This part refers to this [codebase](https://github.com/deeplearning-wisc/cider)

#### In-distribution dataset

We consider the following (in-distribution) datasets: CIFAR and ImageNet. 

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in `./datasets/imagenet/train` and `./datasets/imagenet/val`, respectively.

Please download [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) and place the training data and validation data in `./datasets/CIFAR/CIFAR100` and `./datasets/CIFAR/CIFAR100`, respectively.

#### Out-of-distribution dataset

##### **Small-scale OOD datasets** (for CIFAR)

 For small-scale ID (e.g. CIFAR-10), we use SVHN, Textures (dtd), Places365, LSUN-C (LSUN), LSUN-R (LSUN_resize), and iSUN. 

OOD datasets can be downloaded via the following links (source: [ATOM](https://github.com/jfc43/informative-outlier-mining/blob/master/README.md)):

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/small_OOD_dataset/svhn`. Then run `python utils/select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/small_OOD_dataset/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_datasets/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/small_OOD_dataset/LSUN`.
* [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/small_OOD_dataset/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/small_OOD_dataset/iSUN`.

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd datasets/small_OOD_dataset
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```

The directory structure looks like:

```python
datasets/
---CIFAR10/
---CIFAR100/
---small_OOD_dataset/
------dtd/
------iSUN/
------LSUN/
------LSUN_resize/
------places365/
------SVHN/
```

##### **Large-scale OOD datasets** (for ImageNet)

For large-scale ID (e.g. ImageNet-100), we use the curated 4 OOD datasets from [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), [SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), [Places](http://places2.csail.mit.edu/PAMI_places.pdf), and [Textures](https://arxiv.org/pdf/1311.3618.pdf), and de-duplicated concepts overlapped with ImageNet-1k. The datasets are created by  [Huang et al., 2021](https://github.com/deeplearning-wisc/large_scale_ood) .

The subsampled iNaturalist, SUN, and Places can be downloaded via the following links:

```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz

```
The directory structure looks like this:
```python
datasets/
---ImageNet100/
---ImageNet_OOD_dataset/
------dtd/
------iNaturalist/
------Places/
------SUN/
```


## Training and Evaluation 

### CLIP-based model

Firstly, enter the CLIP-based method folder by running

```
cd CLIP_based/OOD
```

#### Feature extraction

To reduce the training burden, we only fine-tuned a limited number of layers in the pre-trained model. Therefore, we pre-extract features for training, instead of using images to repeatedly extract features with fixed parameters every iteration.

We provide the following script to get the features of CLIP (ViT-B/16):

```
sh feature_extraction_imagenet_100.sh
sh feature_extraction_imagenet_1k.sh
```
We also provide pretrained features for [ImageNet-100](https://drive.google.com/drive/folders/1rkXQYHcaITZCj55OLNXqy_b-yjktONrn?usp=share_link). Since the feature file of ImageNet-1k is too large, we cannot provide it for the time being. Please extract it yourself based on the dataset of ImageNet-1k.
#### Training

We provide sample scripts to train from scratch. Feel free to modify the hyperparameters and training configurations.

```
sh train_npos_imagenet_100.sh
sh train_npos_imagenet_1k.sh
```

#### Evaluation

We use the [MCM](https://openreview.net/forum?id=KnCS9390Va) score as the OOD score to evaluate the fine-tuning CLIP model.

We provide scripts for checkpoint evaluations:

```
sh train_npos_cifar10.sh
sh train_npos_cifar100.sh
sh train_npos_imagenet_100.sh
```


Our checkpoints can be downloaded here for [ImageNet-100](https://drive.google.com/drive/folders/1SjW2kvhDQ6qcsIo5TR7eLMrcL3r6Y3QN?usp=share_link) and [ImageNet-1k](https://drive.google.com/drive/folders/1rkXQYHcaITZCj55OLNXqy_b-yjktONrn?usp=share_link). The performance of these checkpoints is consistent with the results in our paper.

**Evaluate custom checkpoints** 

### **Train from scratch**

Firstly, enter the CLIP-based method folder by running:

```
cd training_from_sctrach
```

#### Training

We provide sample scripts to train from scratch. Feel free to modify the hyperparameters and training configurations.

```
sh train_npos_cifar10.sh
sh train_npos_cifar100.sh
sh train_npos_imagenet_100.sh
```

#### Evaluation

Since methods based on contrastive learning cannot obtain explicit logit output. We use KNN distance as the OOD score to evaluate the fine-tuning CLIP model.

We provide scripts for checkpoint evaluations: 

```
sh test_npos_cifar10.sh
sh test_npos_cifar100.sh
sh test_npos_imagenet_100.sh
```

Our checkpoints can be downloaded here for [ImageNet-100](https://drive.google.com/drive/folders/1SjW2kvhDQ6qcsIo5TR7eLMrcL3r6Y3QN?usp=share_link), [CIFAR-10,](https://drive.google.com/drive/folders/1rkXQYHcaITZCj55OLNXqy_b-yjktONrn?usp=share_link) and [CIFAR-100](https://drive.google.com/drive/folders/1rkXQYHcaITZCj55OLNXqy_b-yjktONrn?usp=share_link). The performance of these checkpoints is consistent with the results in our paper.

### Citation

If you find this work useful in your own research, please cite the paper as:

```
@inproceedings{tao2023nonparametric,
title={Non-parametric Outlier Synthesis},
author={Leitian Tao and Xuefeng Du and Jerry Zhu and Yixuan Li},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=JHklpEZqduQ}
}
```
