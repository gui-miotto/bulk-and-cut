# Bulk and Cut: Multi-objective optimization for CNNs

Bulk and Cut is a tool to jointly optimize accuracy and number of trainable parameters of neural networks.

## Introduction

Bulk and Cut combines a very simple evolutionary strategy together with Bayesian optimization. The name Bulk and Cut comes from the fact that the algorithm first looks for high accuracy models by successively **enlarging** them with [network morphisms][net-morph-paper], then **shrinks** them down using [knowleged distillation][know-dist-paper]. This strategy tries to leverage on the [lotery ticket principle][lot-tick-paper]. That is, large models learn better, nevertheless, once trained, most of the their weights can be dropped without significant accuracy decrease.

This software was first presented to fulfill one of the [requirements](assets/project.pdf) of the course on **Automated Machine Learning 2020**, a colaboration between Uni-Freiburg and Uni-Hannover [AutoML groups][auto-ml-org]. My presentation [slides](assets/Guilherme_Miotto-AutoML2020.odp) can be found in the [assets](assets) directory. Make sure to open them with [LibreOffice Impress][libre-office], otherwise the formatting may be a bit jagged.

Later on, this work was published at the ICML AutoML workshop. The paper is available [here][paper].

## Installation

Bulk and Cut requires python 3.8.

Assuming you have a suitable environment activated, you can install Bulk and Cut with following command:


```sh
python -m pip install git+https://github.com/automl-classroom/final-project-gui-miotto.git
```

## Example

The [examples](examples) folder show how to run Bulk and Cut for different [datasets](datasets). The script [flowers16.py](examples/flowers16.py) shows how run Bulk and Cut on the [project's](assets/project.pdf) official dataset: [micro16flower](datasets/micro16flower). It requires a path to an output directory as an argument. This is the directory where all models, logs and results will be saved.

```sh
python examples/flowers16.py /tmp/bnc_outdir/
```

Bulk and Cut saves many different logs, spreadsheets and plots inside its output directory. For instance, an animated Pareto front is generated:

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
[libre-office]: https://www.libreoffice.org/
[paper]: https://openreview.net/forum?id=yEGlj93aLFY
