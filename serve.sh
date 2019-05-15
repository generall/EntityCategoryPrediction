#!/usr/bin/env bash



MODEL=./data/trained_models/6th_augmented/model.tar.gz \
    FLASK_APP=category_prediction/service/server.py flask run