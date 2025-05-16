# Experiments on the EuLearn Dataset

<!--
## Table of Contents

1. [Description](#description)
2. [Folder Structure](#folder-structure)
3. [Models](#models)
4. [Usage](#usage)

## Description
-->

This repository contains code for deep learning experiments on the **EuLearn dataset**. The dataset can be loaded from either `.pkl` or `.stl` formats. Code is organized into modular folders, each of which contains the necessary scripts to perform the reported experiments.


## Folder Structure

There are three types of models: __attention__, __convolutional__ and __PointNet__-based models, each of which is contained in the corresponding folder.

The **`utils/`** folder contains the following supporting scripts:
- `utils.py` — Includes the implementation of the `NoamOptimizer`, the definition of auxiliary layers for each model, and a function to `visualize` the sampled vertices in a connected graph.
- `train_eval.py`: To train and evaluate the model.
- `dataset.py`: To load the dataset according to its format.


## Models

The following models have been evaluated on the EuLearn dataset:

| Model                              | Script                 | Notes                                 |
| ---------------------------------- | ---------------------- | ------------------------------------- |
| Classic Attention for 3D           | `attention_main.py`    | Multi-head self-attention on 3D inputs|
| Graph Sampled Attention (**ours**) | `gs_attention_main.py` | Uses graph sampling                   |
| Dynamic Graph CNN (DGCNN)          | `dgcnn_main.py`        | Popular for point clouds              |
| Fourier Neural Operator (FNO)      | `fourier_main.py`      | Operator learning                     |
| PointNet                           | `pointnet_main.py`     | Classic point cloud model             |
| PointNet++                         | `pointnetpp_main.py`   | Hierarchical PointNet                 |
| Graph Sampled PointNet (**ours**)  | `gs_pointnet_main.py`  | Combines PointNet with graph sampling |

So each model has a corresponding `[model]_main.py` script that runs the model.


## Usage

To train and test a model, run the following command:
```bash
$ python [model]_main.py --data [folder with training data] --test_data [folder with test data]
```
Additional arguments include:

* `--epochs`: Number of training epochs
* `--dropout`: Dropout rate
* `--d_model`: Model dimensionality
* `--save`: Output name for saving the trained model
* `--load_model`: Input name to load a previously saved model to evaluate or resume training
