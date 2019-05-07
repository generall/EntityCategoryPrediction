#!/usr/bin/env bash


rm -rf EntityCategoryPrediction
git clone https://github.com/generall/EntityCategoryPrediction.git
mkdir -p EntityCategoryPrediction/data


gsutil cp gs://nlp-categories/shrinked_fasttext.model.tgz ./
tar -xzvf shrinked_fasttext.model.tgz

gsutil cp gs://nlp-categories/people_all_categories.json.gz ./
gzip -d people_all_categories.json.gz

gsutil cp gs://nlp-categories/people_mentions.tsv.gz ./
gzip -d people_mentions.tsv.gz

gsutil cp gs://nlp-categories/vocab.tgz ./
tar -xzvf vocab.tgz

ln -f people_all_categories.json EntityCategoryPrediction/data/category_mapping.json

ln -f shrinked_fasttext.model EntityCategoryPrediction/data/fasttext_embedding.model
ln -f shrinked_fasttext.model.vectors_ngrams.npy EntityCategoryPrediction/data/fasttext_embedding.model.vectors_ngrams.npy
ln -f shrinked_fasttext.model.vectors.npy EntityCategoryPrediction/data/fasttext_embedding.model.vectors.npy
ln -f shrinked_fasttext.model.vectors_vocab.npy EntityCategoryPrediction/data/fasttext_embedding.model.vectors_vocab.npy

head -n 2700000 people_mentions.tsv > EntityCategoryPrediction/data/train_data.tsv
tail -n -2700000 people_mentions.tsv | head -n 300000 > EntityCategoryPrediction/data/valid_data.tsv
tail -n -3000000 people_mentions.tsv > EntityCategoryPrediction/data/test_data.tsv

