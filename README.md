# MTLA - Multi-Task Learning Archive

<div align="justify">
Welcome to our Multi-Task Learning Archive Repository! This comprehensive collection houses a diverse range of cutting-edge multi-task learning algorithms, models, datasets, and research implementations. Whether you're a researcher, developer, or enthusiast exploring the realms of machine learning, this repository offers a rich resource pool to delve into multi-task learning techniques across various domains. Explore, experiment, and advance your understanding of simultaneous learning paradigms with our curated collection of resources and tools.
</div>

## Environment Setup
After cloning this repository, ```cd``` inside and use the following commands to create a virtual environment
```
python -m venv .env
source .env/bin/activate
python -m pip install -U pip
```
The requirement packages for this repo are listed below, though ```pip install wheel``` is recommended to run first
```
albumentations==1.3.1
fastparquet==2023.10.1
pandas==2.1.3
tensorboard==2.15.1
tqdm==4.66.1
wandb==0.16.0
scikit-learn==1.3.2
```
The final thing is to install Pytorch (This repository is tested using Ubuntu 22.04, CUDA 12.1, and NVIDIA driver 523s)
```
pip3 install torch torchvision torchaudio
```

## Repository Information
### Current Available Datasets
| Dataset        |  Mode  |   Status  | Related Task                                                                                                                                    |
|----------------|:------:|:---------:|-------------------------------------------------------------------------------------------------------------------------------------------------|
| Oxford Pet III |    -   | Available | Segmentation (3 classes), Classification (37 classes)                                                                                           |
| NYUV2          |    -   | Available | Segmentation (19 classes), Depth Estimation, Surface Normal                                                                                     |
| Cityscape      |  fine  | Available | Segmentation (19 classes), Depth Estimation                                                                                                     |
| Cityscape      | coarse | Available | Segmentation (19 classes), Depth Estimation                                                                                                     |
| CelebA         |    -   | Available | (40+) Attibute Classification (binary labelled), Deep Metric Learning (10k+ identity), Resconstruction (250k+ images), Disentanglement Learning |
### Current Available Methods
| Method                                                                                                                                                                                                      |  Code  |   Status  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------:|:---------:|
| [Gradient Normalization](https://proceedings.mlr.press/v80/chen18a/chen18a.pdf)                                                                                                                             |   gn   | Available |
| [Uncertainty Weighting](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)                                                                       |   uw   | Available |
| [Dynamic Weight Average](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)                                                      |   dwa  | Available |
| [Random Loss Weighting](https://openreview.net/forum?id=jjtFD8A1Wx)                                                                                                                                         |   rlw  | Available |
| [MGDA](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)                                                                                                               |  mgda  | Available |
| [PCGRAD](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)                                                                                                             | pcgrad | Available |
| [CAGRAD](https://openreview.net/forum?id=_61Qh8tULj_)                                                                                                                                                       | cagrad | Available |
| [Recon](https://openreview.net/forum?id=ivwZO-HnzG_)                                                                                                                                                        |  recon |     -     |
| [NashMTL](https://proceedings.mlr.press/v162/navon22a/navon22a.pdf)                                                                                                                                         |  nash  |     -     |
| [Geometric Loss Strategy](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf) |   geo  |     -     |
| [Gradient Sign Dropout](https://papers.nips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html)                                                                                              |   gsd  |     -     |
| [IMTL](https://openreview.net/forum?id=IMPnRXEWpvr)                                                                                                                                                         |  imtl  |     -     |
| [Gradient Vaccine](https://openreview.net/forum?id=F1vEjWK-lH_)                                                                                                                                             |  gvac  |     -     |
| [MoCo](https://openreview.net/forum?id=dLAYGdKTi2)                                                                                                                                                          |  moco  |     -     |
| [Aligned MTL](https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html)                                                   |  amtl  |     -     |
### Current Available Model Architecture
| Based Architecture |          Mode          | Dataset Available |
|:------------------:|:----------------------:|:-----------------:|
| Unet               | Hard Parameter Sharing | OxfordPetIII      |
| SegNet             | Hard Parameter Sharing | OxfordPetIII      |

### Training
Using the parameter in ```main.py``` to perform a customized training process. The experiment evaluation (i.e. loss value, metrics value) is recorded by toggling ```--log``` and ```--wandb```. 
