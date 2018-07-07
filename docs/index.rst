.. BigNmf documentation master file, created by
   sphinx-quickstart on Sat Jul  7 11:32:05 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BigNmf's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

BigNmf
======

BigNmf (Big Data NMF) is a python package for performing single NMF and
joint NMF algorithms. `NMF`_ (Non-negative matrix factorization) is a
unsupervised classification algorithm.

Installation
------------

This package is available on the PyPi repository. Therefore you can
install, by running the following.

.. code:: bash

   pip3 install bignmf

Usage
-----

The following is an example code snippet for running the nmf.

1. Single NMF
~~~~~~~~~~~~~

.. code:: python

   from bignmf.datasets.datasets import Datasets
   from bignmf.models.snmf.standard import StandardNmf

   Datasets.list_all()
   data=Datasets.read("SimulatedX1")
   k = 3
   iter =100
   trials = 50

   model = StandardNmf(data,k)
   model.run(trials, iter, verbose=0)
   print(model.error)
   model.cluster_data()
   model.calc_consensus_matrices()
   print(model.h_cluster)

2. Joint NMF
~~~~~~~~~~~~

.. code:: python

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
   model = StandardNmf(data,k)
   model.run(trials, iter, verbose=0)
   print(model.error)
   model.cluster_data()
   model.calc_consensus_matrices()
   print(model.h_cluster)

.. _NMF: https://en.wikipedia.org/wiki/Non-negative_matrix_factorization