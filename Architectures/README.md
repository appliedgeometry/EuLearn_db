# Experiments on the Eulearn Dataset


## Table of Contents

1. [Description](#description)
2. [Repository Structure](#repository-structure)
3. [Usage](#usage)
4. [Models](#models)


## Description

This repository contains code for deep learning experiments on the **EuLearn dataset**. The dataset can be loaded from either `.pkl` or `.stl` formats. Code is organized into modular folders, each of which contains the necessary scripts to perform the reported experiments.


## Repository Structure

The repository is divided into the following folders:

* `utils`: Contains secondary functions, including the implementation of the NoamOptimizer and auxiliary layers for the deep learning architecture. It also includes scripts for training and evaluating models with the dataset.
	+ `train_eval.py`: Script to train and evaluate models with the dataset.
	+ `dataset.py`: Script containing functions to open the dataset according to its format.

Each directory contains the following:

- **`utils/`** — Supporting code, including the implementation of the `NoamOptimizer` and the definition of auxiliary layers for each model, and the following scripts:
  - `train_eval.py`: To train and evaluate the model.
  - `dataset.py`: To load the dataset according to its format.

Each model script is located in the root or a model-specific folder, named `[model]_main.py`.



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

The following deep learning models were used to test the Eulearn dataset:

1. **Classic Attention for 3D**: `attention_main.py`
2. **Graph Sampled Attention (ours)**: `gs_attention_main.py`
3. **Dynamic Graph Convolutional Neural Network (DGCNN)**: `dgcnn_main.py`
4. **Fourier Neural Operator (FNO)**: `fourier_main.py`
5. **PointNet**: `pointnet_main.py`
6. **PointNet++**: `pointnetpp_main.py`
7. **Graph Sampled PointNet (ours)**: `gs_pointnet_main.py`

Each model has a corresponding script `[model]_main.py` that executes the model.

| Model                              | Script                 | Notes                                 |
| ---------------------------------- | ---------------------- | ------------------------------------- |
| Classic Attention for 3D           | `attention_main.py`    |                                       |
| Graph Sampled Attention (**ours**) | `gs_attention_main.py` | Uses graph sampling                   |
| Dynamic Graph CNN (DGCNN)          | `dgcnn_main.py`        | Popular for point clouds              |
| Fourier Neural Operator (FNO)      | `fourier_main.py`      | Operator learning                     |
| PointNet                           | `pointnet_main.py`     | Classic point cloud model             |
| PointNet++                         | `pointnetpp_main.py`   | Hierarchical PointNet                 |
| Graph Sampled PointNet (**ours**)  | `gs_pointnet_main.py`  | Combines PointNet with graph sampling |
