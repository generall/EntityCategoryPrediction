# Category prediction model

This repo contains AllenNLP model for prediction of Named Entity categories by its mentions.

# Data

## Fake data

You can generate some fake data using this [Notebook](notebooks/gen_face_data.ipynb)


## Real data (Work in progress)

Filtered [OneShotWikilinks](https://www.kaggle.com/generall/oneshotwikilinks) dataset with manually selected categories.

### Data preparation steps


* Crete category graph [build_category_graph.ipynb](./notebooks/build_category_graph.ipynb)
    * Produces: `category_graph.pkl`
* Obtain the list of Person articles from Ontology [obtain_people_articles.ipynb](/notebooks/obtain_people_articles.ipynb):
    * Requires: `dbpedia_2016-10.owl`
    * Produces: `people_categories.json`
* Build mapping from article to people categories [generate_full_people_categories.ipynb](./notebooks/generate_full_people_categories.ipynb). Requires
    * `people_categories.json`
    * `category_graph.pkl`
    * `projects/categories_prediction/manual_categories.gsheet`
* Filter mentions for people [filter_mentions.ipynb](./notebooks/filter_mentions.ipynb). 
    * Requires: `people_all_categories.json`
    * Produces: `people_mentions.tsv`


Prepare splitted data with:

```bash
!split -n l/10 --verbose ../data/fake_data_train.tsv ../data/fake_data_train.tsv_
```

# Install

```bash
pip install -r requirements.txt
```

# Run


## Train

```bash

rm -rf ./data/vocabulary ; allennlp make-vocab -s ./data/ allen_conf_vocab.json --include-package category_prediction

allennlp train -f -s data/stats allen_conf.json --include-package category_prediction
```

```bash
allennlp train -f -s data/stats allen_conf.json --include-package category_prediction -o '{"trainer": {"cuda_device": 0}}'
```

### Continue training with different params

```bash
rm -rf data/stats2/  # Clear new serialization dir
allennlp fine-tune -s data/stats2/ -c allen_conf.json -m ./data/stats/model.tar.gz --include-package category_prediction -o '{"trainer": {"cuda_device": 0}, "iterator": {"base_iterator": {"batch_size": 64}}}'
```

## Validate

```bash
allennlp evaluate ./data/stats/model.tar.gz ./data/fake_data_test.tsv --include-package category_prediction
```

## Server

### Debug

```bash
MODEL=./data/trained_models/6th_augmented/model.tar.gz python run_server.py
```

### Prod

```bash
gunicorn -c gunicorn_config.py wsgi:application
```

### Docker


Build
```bash
cd docker
docker build --tag mention .
```

Run with passing pyenv into container

```bash
docker run --rm --restart unless-stopped -v $HOME:$HOME -p 8000:8000 \
        -v $HOME/.pyenv:/root/.pyenv \ 
        -e ENV_PATH=$HOME/virtualenv/path \
        -e APP_PATH=$HOME/project/root/path mention
```

# GCE related notes


Fix 100% GPU utilization
```bash
sudo nvidia-smi -pm 1
```
