# BigNmf
[![Build Status](https://travis-ci.org/thenmf/bignmf.svg?branch=master)](https://travis-ci.org/thenmf/bignmf)
[![Read the Docs](https://readthedocs.org/projects/bignmf/badge/?version=latest)](https://bignmf.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/bignmf.svg)](https://badge.fury.io/py/bignmf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

BigNmf (Big Data NMF) is a python 3 package for conducting analysis using NMF algorithms.

## NMF Introduction 
[NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)   (Non-negative matrix factorization) factorizes a non-negative input matrix into non-negative factors. The algorithm has an inherent clustering property and has been gaining attention in various fields especially in biological data analysis. 

_Brunet et al_ in their [paper](http://www.pnas.org/content/101/12/4164) demonstrated NMF's superior capability in clustering the [leukemia dataset](https://www.kaggle.com/crawford/gene-expression) compared to standard clustering algorithms like Hierarchial clustering and Self-organizeing maps.

## Available algorithms
The following are the algorithms currently available. If you would like to know more about the algorithm, the links below lead to their papers of origin.
* Single NMF
    1. [Standard Single NMF](https://www.nature.com/articles/44565)
    1. [Sparse NMF](https://www.merl.com/publications/docs/TR2015-023.pdf)
* Joint NMF
    1. [Standard Joint NMF](https://www.ncbi.nlm.nih.gov/pubmed/25411328)
    2. [Integrative NMF](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0176278)

## Installation

This package is available on the PyPi repository. Therefore you can install, by running the following.

```bash
pip3 install bignmf
```

## Usage
The following examples illustrate typical usage of the algorithm.

### 1. Single NMF

```python
from bignmf.datasets.datasets import Datasets
from bignmf.models.snmf.standard import StandardNmf

Datasets.list_all()
data=Datasets.read("SimulatedX1")
k = 3
iter =100
trials = 50

model = StandardNmf(data,k)

# Runs the model
model.run(trials, iter, verbose=0)
print(model.error)

# Clusters the data
model.cluster_data()
print(model.h_cluster)

#Calculates the consensus matrices
model.calc_consensus_matrices() 
print(model.consensus_matrix_w)
```

### 2. Joint NMF

```python
from bignmf.models.jnmf.integrative import IntegrativeJnmf
from bignmf.datasets.datasets import Datasets

Datasets.list_all()
data_dict = {}
data_dict["sim1"] = Datasets.read("SimulatedX1")
data_dict["sim2"] = Datasets.read("SimulatedX2")

k = 3
iter =100
trials = 50
lamb = 0.1

model = IntegrativeJnmf(data_dict, k, lamb)
# Runs the model
model.run(trials, iter, verbose=0)
print(model.error)

# Clusters the data
model.cluster_data()
print(model.h_cluster)

#Calculates the consensus matrices
model.calc_consensus_matrices() 
print(model.consensus_matrix_w)
```

[Here](https://bignmf.readthedocs.io/en/latest/) is the extensive documentation for more details.
