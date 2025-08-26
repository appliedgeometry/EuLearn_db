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

We include three types of models: 

  1. __Attentional__
  2. __Convolutional__
  3. __PointNet-based__

The `utils/` folder provides shared utilities and includes the following scripts:

- `dataset.py` — Loads the dataset based on its format (`.pkl` or `.stl`).
- `train_eval.py` — Contains training and evaluation logic for all models.
- `utils.py` — Implements `NoamOptimizer`, the `LayerNorm` reusable component, and a function to `visualize` sampled vertices as a connected graph.


## Models

The following models have been trained and evaluated with the EuLearn dataset:

| Model                              | Description                           | Script                 | 
| ---------------------------------- | ------------------------------------- | ---------------------- |
| Classic Attention for 3D           | Multi-head self-attention on 3D inputs| `attention_main.py`    |
| Graph Sampled Attention (**ours**) | Uses graph sampling                   | `gs_attention_main.py` |
| Dynamic Graph CNN (DGCNN)          | Popular for point clouds              | `dgcnn_main.py`        |
| Fourier Neural Operator (FNO)      | Operator learning                     | `fourier_main.py`      |
| PointNet                           | Classic point cloud model             | `pointnet_main.py`     |
| PointNet++                         | Hierarchical PointNet                 | `pointnetpp_main.py`   |
| Graph Sampled PointNet (**ours**)  | Combines PointNet with graph sampling | `gs_pointnet_main.py`  |

Run the `[model]_main.py` script to train the model.


## Usage

To train and test a model, from the root directory use:
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

