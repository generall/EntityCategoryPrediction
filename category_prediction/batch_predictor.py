import json
import os
import random
import sys
from pprint import pprint

import category_prediction

import overrides
from allennlp.common import JsonDict, Params
from allennlp.data import Instance
from allennlp.models import load_archive
from allennlp.predictors import Predictor

from category_prediction.settings import DATA_DIR


@Predictor.register('batch-predictor')
class PersonPredictor(Predictor):
    """Predictor wrapper for the EntityCategoryPrediction"""
    pass


if __name__ == '__main__':
    model_path = sys.argv[1]
    input_path = sys.argv[2]

    archive = load_archive(model_path)

    predictor = Predictor.from_archive(archive, 'batch-predictor')

    labels = archive.model.vocab.get_index_to_token_vocabulary("labels")

    for inst in predictor._dataset_reader.read(input_path):
        result = predictor.predict_instance(inst)
        predicted_labels = dict((labels[idx], prob) for idx, prob in enumerate(result['predictions']) if prob > 0.5)
        print(predicted_labels, inst.fields['categories'])
