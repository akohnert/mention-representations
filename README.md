# Exploring Mention Representations for Coreference Resolution in Dialogue

This repository contains the code and scripts needed for the implementation of the experiments described in my Master's Thesis "Exploring Mention Representations for Coreference Resolution in Dialogue" (submitted 18/09/2023).

The code in *coref-hoi* is based on the model from [this paper](https://aclanthology.org/2020.emnlp-main.686.pdf), with the code from [this repository](https://github.com/lxucs/coref-hoi/).


## Setup

This describes the setup for training and prediction, see folder `coref-hoi`.

* Create a Python3 environment (I used Python 3.9.2)
* Install the dependencies: `pip install -r requirements.txt`
* Create a data directory that will contain all data, model and log files; adjust the paths in  `experiments.conf`
* If training on OntoNotes 5.0, prepare the dataset: `./setup_data.sh /path/to/ontonotes /path/to/data/dir`


For SpanBERT, download the pretrained weights from the base model from [this repository](https://github.com/facebookresearch/SpanBERT), and rename it /path/to/data/dir/spanbert_base 

When training with additional embeddings, you need need to download the GloVe and Numberbatch embeddings:
* __Glove__: 300-dimensional vectors via `!wget http://nlp.stanford.edu/data/glove.6B.zip`
* __Numberbatch__: 300-dimensional vectors via `numberbatch-en-19.08.txt.gz` from [here](https://github.com/commonsense/conceptnet-numberbatch).


## Data Preparation

The scripts to transfrom json to UA-format files and vice versa are taken and adjusted from [this repository](https://github.com/sopankhosla/codi2021_scripts).

Helpful functions for recreating the results from the thesis are included in `custom_input.py`:

* `ua_to_json`: Transform conll-UA files into json files
* `json_to_ua`: Transform json files into conll-UA files
* `json_remove_singleton`: Remove the singleton mentions/clusters from json files
* `pred_json_to_ua`: Prepare an output json file (with predictions) for evaluation with a conll-UA scorer
* `get_stats`: Show some numbers about a json file, e.g. number mentions and clusters
* `combine_corpora`: Combine several json files of different corpora into one large json file


## Training

`python run.py [config] [gpu_id]`

* [config] can be any configuration in experiments.conf. This thesis used three configurations, described in the next section.
* Log file will be saved at your_data_dir/[config]/log_XXX.txt
* Models will be saved at your_data_dir/[config]/model_XXX.bin
* Tensorboard is available at your_data_dir/tensorboard


## Configurations

| Config           | Description |
| :----------- | :------ |
| train_spanbert_base_singleton | A SpanBert base model that predicts singletons (the thesis' baseline) |
| bert_base_singleton           | As above, but uses a Bert model instead |
| train_spanbert_base_ml0_d1    | A SpanBert base model that does __not__ predict singletons |


## Evaluation

`python ua-scorer.py [key.conllua] [prediction.conllua] [remove_singletons] [remove_split_antecedent]`

The code for The Universal Anaphora Scorer is taken from [this repository](https://github.com/juntaoy/universal-anaphora-scorer). The arguments `[remove_singletons] [remove_split_antecedent]` must be included when the singletons in the data shall be ignored (and not taken into account for scoring).





