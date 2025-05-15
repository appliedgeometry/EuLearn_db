# Experiments for the Eulearn Dataset


## Table of Contents

1. [Description](#description)
2. [Repository Structure](#repository-structure)
3. [Usage](#usage)
4. [Models](#models)


## Description

This repository contains code for experiments on the Eulearn Dataset. The dataset can be loaded from pickle or stl formats. The code is organized into folders, each containing necessary scripts to replicate the experiments reported.


## Repository Structure

The repository is divided into the following folders:

* `utils`: Contains secondary functions, including the implementation of the NoamOptimizer and auxiliary layers for the deep learning architecture. It also includes scripts for training and evaluating models with the dataset.
	+ `train_eval.py`: Script to train and evaluate models with the dataset.
	+ `dataset.py`: Script containing functions to open the dataset according to its format.


## Usage

To execute a model, run the following command:
```bash
$ python [model]_main.py --data [folder with training data] --test_data [folder with test data]
```
Additional arguments include:

* `--epochs`: Number of epochs
* `--dropout`: Dropout value
* `--d_model`: Dimension of the model
* `--save`: Name to save the learning model
* `--load_model`: Name of a previously saved model (if exists)


## Models

The following deep learning models are used to test the Eulearn dataset:

1. **Classic Attention for 3d**: `attention_main.py`
2. **Graph Sampled Attention (ours)**: `gs_attention_main.py`
3. **Dynamic Graph Convolutional Neural Network (DGCNN)**: `dgcnn_main.py`
4. **Fourier Neural Operator (FNO)**: `fourier_main.py`
5. **PointNet**: `pointnet_main.py`
6. **PointNet++**: `pointnetpp_main.py`
7. **Graph Sampled PointNet (ours)**: `gs_pointnet_main.py`

Each model has a corresponding script `[model]_main.py` that executes the model.
