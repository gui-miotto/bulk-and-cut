# Bulk and Cut: Multi-objective optimization for CNNs

Bulk and Cut is a tool to jointly optimize accuracy and number of trainable parameters of neural networks. More specifically, convolutional neural network (CNN) classifiers.

## Introduction

Bulk and Cut combines a very simple evolutionary strategy with Bayesian optimization. The name Bulk and Cut comes from the fact the algorithm first looks for high accuracy models by succesvelly **enlarging** them with [network morphisms][net-morph-paper], then **shrinks** them down using [knowleged distillation][know-dist-paper]. This strategy tries to leverage on the [lotery ticket principle][lot-tick-paper]: large models learn better, nevertheless, once trained, most of the their weights can be dropped without signifcant accuracy loss.

This work is my effort to fulfill one of the [requirements](assets/project.pdf) of the course on **Automated Machine Learning 2020**, a colaboration between Uni-Freiburg and Uni-Hannover.

## Repo structure
* [micro17flowers](micro17flowers) <BR>
  contains a downsampled version of a [flower classification dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).
  We cropped and resized it such that the resulting images are of size 16x16.

* [src](src) Source folder
    * [baseline_comparisons.ipynb](src/baseline_comparisons.ipynb) <BR>
      contains these baselines aswell as a simple plotting script you should use when presenting your final results.

    * baseline_values.pkl<BR>
      contains the baseline pareto_front

    * [bohb.py](src/bohb.py) <BR>
      contains a simple example of how to use BOHB on the dataset. This implementation is single objective only!
      This does **not** minimize the network size!

    * [cnn.py](src/cnn.py)<BR>
      contains the source code of the network you need to optimize. It optimizes the top-3 accuracy.

    * [main.py](src/main.py)<BR>
      contains an example script that shows you how to instantiate the network and how to evaluate its 3-fold stratified
      CV accuracy as well as its size. This file also gives you the **default configuration** that always has to be in your
      serch space.

    * [util.py](src/util.py)<BR>
      contains simple helper functions for cnn.py


<!-- Markdown link & img dfn's -->
[net-morph-paper]: https://arxiv.org/abs/1511.05641
[know-dist-paper]: https://arxiv.org/abs/1503.02531
[lot-tick-paper]: https://arxiv.org/abs/1803.03635