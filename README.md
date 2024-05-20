# DiffPath

This is the codebase for Out-of-Distribution Detection with a Single Unconditional Diffusion Model (DiffPath), implemented using PyTorch. The codebase is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

## Installation

It is recommended to install dependencies in a ```conda``` environment:
```
conda create --name diffpath python=3.8
pip install -r requirements.txt
```

## Download Diffusion Model Checkpoint
Download the ImageNet-64 diffusion model checkpoint trained with ```L-hybrid``` objective from the [openai/improved-diffusion](https://github.com/openai/improved-diffusion) repo. The link is provided here as well:
[https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt).

## OOD Detection with DiffPath
To perform OOD detection with DiffPath, first we calculate the diffusion path statistics for both train and test sets. We demonstrate the steps for the task of CIFAR10 (in-dist) vs SVHN (out-of-dist). 
```
# calculate statistics for CIFAR10 training set on GPU ID 0
python save_train_statistics.py --data_dir /path/to/cifar10/dataset --dataset cifar10 --model_path /path/to/imagenet64/model/checkpoint
--batch_size 256 --n_ddim_steps 50 --device 0
```
```
# calculate statistics for CIFAR10 test set on GPU ID 0
python save_test_statistics.py --data_dir /path/to/cifar10/dataset --dataset cifar10 --model_path /path/to/imagenet64/model/checkpoint
--batch_size 256 --n_ddim_steps 50 --device 0
```
```
# calculate statistics for SVHN test set on GPU ID 0
python save_test_statistics.py --data_dir /path/to/svhn/dataset --dataset svhn --model_path /path/to/imagenet64/model/checkpoint
--batch_size 256 --n_ddim_steps 50 --device 0
```
The statistics will be saved as ```.npz``` files in ```train_statistics/ddim50``` and ```test_statistics/ddim50``` respectively.

Now perform OOD detection using DiffPath-6D:
```
python eval_6d.py --in_dist cifar10 --out_of_dist svhn --n_ddim_steps 50
```
The results will be printed to the screen.

