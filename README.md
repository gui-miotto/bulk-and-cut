# AutoML lecture 2020 (Freiburg & Hannover)
## Final Project

This repository contains all things needed for the final project.
Your task is to jointly optimize a networks accuracy (maximize) and size (minimize),
such that you produce a set of solution candidates on a pareto front.
We provide [two baselines](src/baseline_comparisons.ipynb). A rather simple baseline and one that was minimally optimized
by Difan & André.

## Repo structure
* [micro17flowers](micro17flowers) <BR>
  contains a downsampled version of a [flower classification dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html).
  We cropped ans resized it such that the resulting images are of size 16x16.

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