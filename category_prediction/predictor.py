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


@Predictor.register('person-predictor')
class PersonPredictor(Predictor):
    """Predictor wrapper for the EntityCategoryPrediction"""

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        mentions = json_dict['mentions']
        mentions = random.choices(
            mentions, k=self._dataset_reader.sentence_sample)

        instance = self._dataset_reader.text_to_instance(sentences=mentions)
        return instance


if __name__ == '__main__':
    model_path = sys.argv[1]

    overrides = json.dumps({
        "dataset_reader": {
            "category_mapping_file": None,
            "token_indexers": {
                "tokens": "single_id"
            }
        },
        "model": {
            "text_embedder": {
                "embedder_to_indexer_map": {
                    "tokens-ngram": ["tokens"],
                    "tokens": ["tokens"],
                },
                "token_embedders": {
                    "tokens-ngram": {
                        "type": "disk-gensim-embedder",
                        "model_path": os.path.join(DATA_DIR, "fasttext_embedding.model"),
                        "model_params_path": os.path.join(DATA_DIR, "fasttext_embedding.model.params"),
                        "dimensions": 300
                    }
                }
            }
        }
    })

    archive = load_archive(model_path, overrides=overrides)

    predictor = Predictor.from_archive(archive, 'person-predictor')

    result = predictor.predict_json({
        "mentions": [
            "@@mention@@ is a mathematician",
            "Millennium Prize Problem was solved by @@mention@@ in 2002",
        ]
    })

    labels = archive.model.vocab.get_index_to_token_vocabulary("labels")

    result.items()

    predicted_labels = dict((labels[idx], prob) for idx, prob in enumerate(
        result['predictions']) if prob > 0.5)

    pprint(predicted_labels)
