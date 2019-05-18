import json
import os
import random
import sys
from pprint import pprint
import numpy as np

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

    overrides = json.dumps({
        "dataset_reader": {
            "category_mapping_file": None,
            "token_indexers": {
                "tokens": "single_id"
            }
        },
        "model": {
            "seq_combiner": {
                "return_weights": True
            },
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

    @classmethod
    def select_mentions(cls, mentions, sample_size):

        if len(mentions) > sample_size:
            mentions = random.sample(mentions, sample_size)
        elif len(mentions) < sample_size:
            mentions += random.choices(mentions, k=sample_size - len(mentions))

        return mentions

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        mentions = json_dict['mentions']

        assert len(mentions) == self._dataset_reader.sentence_sample
        instance = self._dataset_reader.text_to_instance(sentences=mentions)
        return instance


if __name__ == '__main__':
    model_path = sys.argv[1]

    archive = load_archive(model_path, overrides=PersonPredictor.overrides)

    predictor = Predictor.from_archive(archive, 'person-predictor')

    result = predictor.predict_json({
        "mentions": PersonPredictor.select_mentions([
            "@@mb@@ Perelman @@me@@ is Russian writer",
            "Millennium Prize Problem was solved by @@mb@@ him @@me@@ in 1998 and then he died",
        ], predictor._dataset_reader.sentence_sample)
    })

    labels = archive.model.vocab.get_index_to_token_vocabulary("labels")

    predicted_labels = dict((labels[idx], prob) for idx, prob in enumerate(
        result['predictions']) if prob > 0.5)

    pprint(predicted_labels)

    # sample size, num_layers, 1, num_heads, num_words
    print(np.array(result['attention_weights']).shape)
