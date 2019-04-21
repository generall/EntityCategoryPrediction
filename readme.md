# Category prediction model (Work in progress)

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

## Validate

```bash
allennlp evaluate ./data/stats/model.tar.gz ./data/fake_data_test.tsv --include-package category_prediction
```

