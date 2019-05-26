import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Tuple

import numpy as np
import spacy
from allennlp.models import load_archive
from allennlp.predictors import Predictor

from category_prediction.predictor import PersonPredictor
from category_prediction.settings import DATA_DIR


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


class MentionExtractor:

    def __init__(self, model_path, context_size=100, label='PERSON'):
        self.label = label
        self.context_size = context_size
        self.ner = spacy.load("en_core_web_sm")
        self.archive = load_archive(model_path, overrides=PersonPredictor.overrides)
        self.predictor: Predictor = Predictor.from_archive(self.archive, 'person-predictor')

        self.dataset_reader = self.predictor._dataset_reader

    def extract_mentions(self, text):
        text = text.replace("\n", ' ')

        doc = self.ner(text)

        mentions = []

        for ent in doc.ents:
            if ent.label_ == self.label:
                mention = ent.text

                left_context = text[max(ent.start_char - self.context_size, 0):ent.start_char]
                right_context = text[ent.end_char:ent.end_char + self.context_size]

                mentions.append({
                    "left_context": left_context,
                    "right_context": right_context,
                    "mention": mention
                })

        return mentions

    @classmethod
    def get_mention_words(cls, mention) -> dict:
        return dict(
            map(
                lambda x: (x[1], x[0]),
                filter(
                    lambda x: len(x[1]) > 2,
                    enumerate(re.split(r'\W+', mention.lower()))
                )
            )
        )

    def group_mentions(self, mentions: List[dict]) -> List[dict]:
        groups = []

        for mention in mentions:

            mention_words = self.get_mention_words(mention['mention'])

            is_found = False
            for group in groups:
                group_words = group['words']

                if len(set(mention_words).intersection(set(group_words))) > 0:
                    is_found = True
                    group['mentions'].append(mention)
                    group_words.update(mention_words)
                    break

            if not is_found:
                groups.append({
                    'words': mention_words,
                    'mentions': [mention]
                })

        for group in groups:
            group['words'] = list(sorted(group['words'], key=lambda word: group['words'][word]))

        return groups

    def merge_sentence(self, mention):
        return f"{mention['left_context']} {self.dataset_reader.left_tag} {mention['mention']}" \
            f" {self.dataset_reader.right_tag} {mention['right_context']}"

    def restore_attention(
            self,
            attention_mapping: Dict[int, List[np.ndarray]],
            sentences: List[str]
    ) -> Dict[int, List[Tuple[str, list]]]:
        """

        :param attention_mapping: sent_id => attentions [num_layers * 1 * num_heads * num_words]
        :param sentences:
        :return:
        """
        res = {}

        for idx, attentions in attention_mapping.items():
            sentence = sentences[idx]
            tokens = self.dataset_reader.tokenizer.tokenize(sentence)
            num_tokens = len(tokens)

            # [num_tokens * num_layers * num_heads]
            attentions = np.mean(
                np.stack([np.transpose(np.array(attention), [1, 3, 0, 2])[0][:num_tokens] for attention in attentions]),
                axis=0)

            tokens_weights = []
            for token, weights in zip((token.text for token in tokens), attentions.tolist()):
                tokens_weights.append({
                    'token': token,
                    'weights': weights
                })

            res[idx] = tokens_weights

        return res

    def predict_group(self, group):

        sentences = [self.merge_sentence(mention) for mention in group]

        subgroups_ids = chunker(
            list(range(len(group))),
            self.dataset_reader.sentence_sample
        )

        subgroups_ids = [PersonPredictor.select_mentions(x, self.dataset_reader.sentence_sample) for x in subgroups_ids]

        json_instances = []
        for subgroup_ids in subgroups_ids:
            subgroup = [sentences[idx] for idx in subgroup_ids]
            json_instances.append({'mentions': subgroup})

        predictions = self.predictor.predict_batch_json(json_instances)

        scores = None

        attention_mapping = defaultdict(list)

        for prediction, subgroup_ids in zip(predictions, subgroups_ids):
            if scores is None:
                scores = np.array(prediction['predictions'])
            else:
                scores += np.array(prediction['predictions'])

            for mention_id, attention in zip(subgroup_ids, prediction['attention_weights']):
                attention_mapping[mention_id].append(np.array(attention))

        attention_mapping = self.restore_attention(attention_mapping, sentences)

        scores = scores / len(predictions)
        labels = self.archive.model.vocab.get_index_to_token_vocabulary("labels")
        predicted_labels = dict((labels[idx], prob) for idx, prob in enumerate(scores) if prob > 0.2)

        return {
            "labels": predicted_labels,
            "attention": attention_mapping
        }

    @lru_cache(maxsize=32)
    def extract_and_predict(self, text):
        mentions = self.extract_mentions(text)
        mention_groups = self.group_mentions(mentions)

        return [{
            "mention": mention_group['words'],
            "prediction": self.predict_group(mention_group['mentions'])
        } for mention_group in mention_groups]


if __name__ == '__main__':
    me = MentionExtractor(os.path.join(DATA_DIR, 'trained_models/6th_augmented/model.tar.gz'))

    text = """
Robert Pattinson is best known for portraying Edward Cullen in the film adaptations of author Stephanie Meyer's Twilight series.
    """

    mentions = me.extract_mentions(text)

    mention_groups = me.group_mentions(mentions)
    res = me.predict_group(mention_groups[0]['mentions'])

    labels, attention_mapping = res['labels'], res['attention']

    print(json.dumps(mention_groups, indent=4))
    print(json.dumps(labels, indent=4))
