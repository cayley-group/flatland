# Flatland

![Test status](https://github.com/cayley-group/flatland/workflows/Test/badge.svg?branch=master)
[![DOI](https://zenodo.org/badge/337879868.svg)](https://zenodo.org/badge/latestdoi/337879868)

[Strategy Context Whitepaper](https://docs.google.com/document/d/1cMVCeON0DSIUaUSnoD4yAZSC1s0sEwxZnBiTgh8gbkk/edit?usp=sharing) | [Manuscript Working Doc](https://docs.google.com/document/d/19KuoO6f2GiGr6688aCDbZatNUPUOVvEtg0HgTaTYMbE/edit?usp=sharing)

#### Easier development of polymer structure prediction tools

The development and debugging of modern neural network approaches for protein structure prediction has the potential to be a somewhat complicated task. Here we provide a framework, Flatland, for simulating training datasets of the form used by current state-of-the-art systems. The central task performed is to evolve a population of polymer sequences, predict their folded structure, produce distance matrix and bond angle tensors, and calculate spectra of per-polymer compound or other polymer interaction likelihoods (emulating protein-protein and compound-protein interaction datasets). This framework intentionally performs such simulations in a highly simplified manner that does not accurately model the real biological processes. Relatedly, it provides the means to simplify the problem to an arbitrary extent - including permitting the reduction from three to two dimensions, reducing the amino acid alphabet from twenty-one to three, and simplifying characteristics of the evolutionary optimization. We provide the framework together with a collection of starting-point notebooks for training neural network models to learn from such data to kick-start the development efforts of those interested in contributing to solving this vitally important problem.

## Installation

You can install the last released version of Flatland via pip

```bash

pip install cg-flatland

```

Or install the most recent development version of Flatland using git

```bash

git clone https://github.com/cayley-group/flatland
pip install -e flatland

```

## Demonstrations

We're putting together various demonstration notebooks which you'll be able to access via the following:

* [Data Generation](https://colab.research.google.com/github/cayley-group/flatland/blob/master/nb/data-generation.ipynb)
* [Structure Solver Example](https://colab.research.google.com/github/cayley-group/flatland/blob/master/nb/structure-solver.ipynb)

As this project is for the purpose of enhancing people's ability to make progress in this problem area we do welcome requests and comments via [GitHub Issues](https://github.com/cayley-group/flatland/issues) for ways to enhance these in that regard (including of course bugs).

### Citing this work

This work will be published shortly via ArXiv then in a peer-reviewed venue. Until then, you can cite the Flatland environment using the following:

```
@misc{flatland2021,
    title={Flatland: A simulation environment for simplifying the development of polymer structure prediction tools.},
    author={Christopher Beitel},
    year={2021},
    howpublished={\url{https://github.com/cayley-group/flatland}, \url{https://doi.org/10.5281/zenodo.4536540}},
    archivePrefix={Zenodo},
}
```

The same applies to the work-in-progress strategic review linked above:

```
@misc{ngbiophysicsbrainstorm2021,
    title={Brainstorming next-generation methods for biophysical simulation and biopolymer design.},
    author={Christopher Beitel},
    year={2021},
    howpublished={\url{https://github.com/cayley-group/pub-review}, \url{https://doi.org/10.5281/zenodo.4536538}},
    archivePrefix={Zenodo},
}
```
