#!/usr/bin/env bash


cd category_prediction/service
npm run build
cd ../../


rsync -avP --exclude='web_env' --exclude='data' --exclude='__pycache__' --exclude='node_modules' . $HOST:./projects/EntityCategoryPrediction

rsync -avPL data/fasttext_embedding.model* $HOST:./projects/EntityCategoryPrediction/data/

rsync -avPL data/trained_models/ $HOST:./projects/EntityCategoryPrediction/data/trained_models


#ssh $HOST 'python -m spacy download en_core_web_sm'