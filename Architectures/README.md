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

We include three types of models: __attention__, __convolutional__ and __PointNet-based__.

The `utils/` folder provides shared utilities and includes the following scripts:

- `dataset.py` — Loads the dataset based on its format (`.pkl` or `.stl`).
- `train_eval.py` — Contains training and evaluation logic for all models.
- `utils.py` — Implements `NoamOptimizer`, the `LayerNorm` reusable component, and a function to `visualize` sampled vertices as a connected graph.


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

To train and test a model, use:
```bash
python [model]_main.py --data [folder with training data] --test_data [folder with testing data]
```
Additional arguments include:

<!--
* `--epochs`: Number of training epochs
* `--dropout`: Dropout rate
* `--d_model`: Model dimensionality
* `--save`: Output name for saving the trained model
* `--load_model`: Input name to load a previously saved model to evaluate or resume training
-->

| Argument         | Description                                                  |
|------------------|--------------------------------------------------------------|
| `--epochs`       | Number of training epochs                                    |
| `--dropout`      | Dropout rate                                                 |
| `--d_model`      | Model dimensionality                                         |
| `--save`         | Output name for saving the trained model                     |
| `--load_model`   | Input name to load a previously saved model for evaluation or resuming training |

