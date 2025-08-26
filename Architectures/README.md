# EuLearn Dataset Model Training

This folder contains deep learning models that were trained and evaluated with the **EuLearn dataset**. The dataset can be loaded from either `.pkl` or `.stl` formats. 
The code is organized in modular folders, each of which contains the necessary scripts to perform the reported experiments.


## Table of Contents

1. [Models](#models)
2. [Usage](#usage)


## Models

We worked with three types of models: 

  1. __Attentional__
  2. __Convolutional__
  3. __PointNet-based__

The following models were trained and evaluated with the EuLearn dataset:

| Model                              | Description                           | Script                 | 
| ---------------------------------- | ------------------------------------- | ---------------------- |
| Classic Attention for 3D           | Multi-head self-attention on 3D inputs| `attention_main.py`    |
| Graph Sampled Attention (**ours**) | Uses graph sampling                   | `gs_attention_main.py` |
| Dynamic Graph CNN (DGCNN)          | Popular for point clouds              | `dgcnn_main.py`        |
| Fourier Neural Operator (FNO)      | Operator learning                     | `fourier_main.py`      |
| PointNet                           | Classic point cloud model             | `pointnet_main.py`     |
| PointNet++                         | Hierarchical PointNet                 | `pointnetpp_main.py`   |
| Graph Sampled PointNet (**ours**)  | Combines PointNet with graph sampling | `gs_pointnet_main.py`  |

The `utils/` folder provides shared utilities:

- `dataset.py` — Loads the dataset based on its format (`.pkl` or `.stl`).
- `train_eval.py` — Contains training and evaluation logic for all models.
- `utils.py` — Includes `NoamOptimizer`, a `LayerNorm` reusable component, and a function to `visualize` the sampled vertices as a connected graph.

In `[model]_main.py`, replace `from train_eval import ...` and `from dataset import ... ` with:
```
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_eval import train_model, eval_model, train_adj_model, eval_adj_model
from utils.dataset import Dataset, get_data, get_field, get_point_cloud, get_sampled_pointclouds, get_surfaces
```


## Usage

To train and test a model, from the root directory use:
```
python [model folder]/[model]_main.py --data [training data folder] --test_data [testing data folder]
```
Additional arguments include:

<!--
* `--epochs`: Number of training epochs
* `--dropout`: Dropout rate
* `--d_model`: Model dimensionality
* `--save`: Output name for saving the trained model
* `--load_model`: Input name to load a previously saved model to evaluate or resume training
-->

| Argument         | Description                                                                     |
|------------------|---------------------------------------------------------------------------------|
| `--epochs`       | Number of training epochs                                                       |
| `--dropout`      | Dropout rate                                                                    |
| `--d_model`      | Model dimensionality                                                            |
| `--save`         | Output name for saving the trained model                                        |
| `--load_model`   | Input name to load a previously saved model for evaluation or resuming training |
