# Bulk and Cut: Multi-objective optimization for CNNs

Bulk and Cut is a tool to jointly optimize accuracy and number of trainable parameters of neural networks. More specifically, convolutional neural network (CNN) classifiers.

## Introduction

Bulk and Cut combines a very simple evolutionary strategy with Bayesian optimization. The name Bulk and Cut comes from the fact the algorithm first looks for high accuracy models by successively **enlarging** them with [network morphisms][net-morph-paper], then **shrinks** them down using [knowleged distillation][know-dist-paper]. This strategy tries to leverage on the [lotery ticket principle][lot-tick-paper]: large models learn better, nevertheless, once trained, most of the their weights can be dropped without signifcant accuracy loss.

This work is my effort to fulfill one of the [requirements](assets/project.pdf) of the course on **Automated Machine Learning 2020**, a colaboration between Uni-Freiburg and Uni-Hannover [AutoML groups][auto-ml-org].

## Installation

Bulk and Cut requires python>=3.8.

Assuming you have a suitable environment activated, you can install Bulk and Cut with following command:


```sh
python -m pip install git+https://github.com/automl-classroom/final-project-gui-miotto.git
```

## Example

The [examples](examples) folder show how to run Bulk and Cut for different [datasets](datasets). The script [flowers16.py](examples/flowers16.py) shows how run Bulk and Cut on the [project's](assets/project.pdf) official dataset: [micro16flower](datasets/micro16flower). If needed, alter the `output_dir` specified inside `flowers16.py`. This is the directory where all logs and results will be saved. Finally, run it.

```sh
python examples/flowers16.py
```

Bulk and Cut saves many different logs, spreadsheets and plots inside the `output_dir`. For instance, an animation of the evolution of the Pareto front is generated:

![ParetoFront](assets/animated_pareto_front.gif)

## Keywords:

* Convolutional Neural Networks
* Neural Architecture Search
* Hyper-parameter optimization
* Auto-ML
* Evolutionary algorithm
* Bayesian optimization
* Multi-objective optimization
* Pareto optimality
* Non-dominated sorting
* Neral network morphisms
* Neural network pruning
* Knowledge distillation
* Mix-up data augmentation


<!-- Markdown link & img dfn's -->
[net-morph-paper]: https://arxiv.org/abs/1511.05641
[know-dist-paper]: https://arxiv.org/abs/1503.02531
[lot-tick-paper]: https://arxiv.org/abs/1803.03635
[auto-ml-org]: https://www.automl.org/