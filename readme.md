# Category prediction model (Work in progress)

This repo contains AllenNLP model for prediction of Named Entity categories by its mentions.

# Data

## Fake data

You can generate some fake data using this [Notebook](notebooks/gen_face_data.ipynb)


## Real data (Work in progress)

Filtered [OneShotWikilinks](https://www.kaggle.com/generall/oneshotwikilinks) dataset with manually selected categories.


# Install

```bash
pip install -r requirements.txt
```

# Run

```bash
allennlp train -f -s data/stats allen_conf.json --include-package category_prediction
```
