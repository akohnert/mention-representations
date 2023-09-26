# Exploring Mention Representations for Coreference Resolution in Dialogue

This repository contains the code and scripts needed for the implementation of the experiments described in my Master's Thesis "Exploring Mention Representations for Coreference Resolution in Dialogue" (submitted 18/09/2023).

The code in *coref-hoi* is based on the model from [this paper](https://aclanthology.org/2020.emnlp-main.686.pdf), with the code from [this repository](https://github.com/lxucs/coref-hoi/).


## Setup

This describes the setup for training and prediction, see folder `coref-hoi`
* Create an Python3 environment (I used Python 3.9.2)
* Install Python3 dependencies: `pip install -r requirements.txt`
* Create a data directory that will contain all data, model and log files; adjust the paths in  `experiments.conf`
* If training on OntoNotes 5.0, prepare the dataset: `./setup_data.sh /path/to/ontonotes /path/to/data/dir`

