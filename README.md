# Deep Learning CNN

> Introduction and analysis of CNN components, architecture, and deep ensemble learning

## Repository Contents

1. CNN Architecture [[Jupyter Notebook](https://nbviewer.jupyter.org/github/cschupbach/deep_learning_cnn/blob/master/cnn_architecture.ipynb "CNN Architecture Jupyter Notebook")]

2. Digits Ensemble CNN [[Jupyter Notebook](https://nbviewer.jupyter.org/github/cschupbach/deep_learning_cnn/blob/master/digits_ensemble.ipynb "CNN Architecture Jupyter Notebook")]

3. Fashion Ensemble CNN [[Jupyter Notebook](https://nbviewer.jupyter.org/github/cschupbach/deep_learning_cnn/blob/master/fashion_ensemble.ipynb "CNN Architecture Jupyter Notebook")]

## GitHub Conflict
Due to known rendering conflicts (issues [#3555](https://github.com/jupyter/notebook/issues/3555), [#5323](https://github.com/jupyter/notebook/issues/5323)), GitHub may not load `*.ipynb` files correctly; including `*.pdf` files rendered with `nbconvert`. This is a frequent closed-source rendering conflict on GitHub's end. If you receive the following error, please try the links under the [Repository Contents](https://github.com/cschupbach/deep_learning_cnn#repository-contents) section above.

![Rendering Conflict](/fig/figure13.png "Rendering Conflict")

## Quick Overview

### Feature Learning

#### Activated Convolutional Layers

![Convolution + ReLU](/fig/figure02.png "Convolution + ReLU")

#### Max Pooling

![Convolution + ReLU + Max Pool](/fig/figure03.png "Convolution + ReLU + Max Pool")

#### Stacking Layers
![Convolution + . . . + Max Pool](/fig/figure04.png "Convolution + . . . + Max Pool")

### Classification
#### Fully Connected (FC) Layers

![Flatten 1](/fig/figure05.png "Flatten 1")
![Dense 1](/fig/figure06.png "Dense 1")
![Dense 2](/fig/figure07.png "Dense 2")
![Dense 3](/fig/figure08.png "Dense 3")

## Digits Ensemble CNN

> [Jupyter Notebook](https://nbviewer.jupyter.org/github/cschupbach/deep_learning_cnn/blob/master/digits_ensemble.ipynb "CNN Architecture Jupyter Notebook")

**Results Confusion Matrix**
![Digits Ensemble Confusion Matrix](/fig/figure09.png "Digits Ensemble Confusion Matrix")

**Select Misclassified Digits**
![First 10 Misclassified Digits](/fig/figure10.png "First 10 Misclassified Digits")

## Fashion Ensemble CNN

> [Jupyter Notebook](https://nbviewer.jupyter.org/github/cschupbach/deep_learning_cnn/blob/master/fashion_ensemble.ipynb "CNN Architecture Jupyter Notebook")

**Results Confusion Matrix**
![Fashion Ensemble Confusion Matrix](/fig/figure11.png "Fashion Ensemble Confusion Matrix")

**First 10 Misclassified Fashion Items**
![First 10 Misclassified Items](/fig/figure12.png "First 10 Misclassified Items")

---
[![GitHub](https://img.shields.io/github/license/cschupbach/deep_learning_cnn)](https://github.com/cschupbach/deep_learning_cnn/blob/master/LICENSE)