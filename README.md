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
### Current Available Model Architecture
