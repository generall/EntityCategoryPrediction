import json
import os
import re
from collections import defaultdict
from itertools import zip_longest
from pprint import pprint
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
                left_context = text[ent.start_char - self.context_size:ent.start_char]
                right_context = text[ent.end_char:ent.end_char + self.context_size]

                mentions.append({
                    "left_context": left_context,
                    "right_context": right_context,
                    "mention": mention
                })

        return mentions

    @classmethod
    def get_mention_words(cls, mention) -> set:
        return set(filter(lambda x: len(x) > 2, re.split(r'\W+', mention.lower())))

    def group_mentions(self, mentions: List[dict]):
        groups = []

        for mention in mentions:

            mention_words = self.get_mention_words(mention['mention'])

            is_found = False
            for group in groups:
                group_words = group['words']

                if len(mention_words.intersection(group_words)) > 0:
                    is_found = True
                    group['mentions'].append(mention)
                    group_words.update(mention_words)
                    break

            if not is_found:
                groups.append({
                    'words': mention_words,
                    'mentions': [mention]
                })

        return [group['mentions'] for group in groups]

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
            res[idx] = list(zip((token.text for token in tokens), attentions.tolist()))

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

        return predicted_labels, attention_mapping


if __name__ == '__main__':
    me = MentionExtractor(os.path.join(DATA_DIR, 'trained_models/6th_augmented/model.tar.gz'))

    text = """
    
Maisie Williams says there was a "huge period" of her life when "I'd tell myself every day that I hated myself".

The Game of Thrones star says that growing up in the public eye, she felt pressure to pretend "that everything is fine" when actually she wasn't very happy.

"It got to the point where I'd be in a conversation with my friends and my mind would be running and running and running and thinking about all the stupid things I'd said in my life, and all of the people that had looked at me a certain way, and it would just race and race and race," Maisie said on Fearne Cotton's Happy Place podcast.

"I think we can all relate to that - telling ourselves awful things."

Image caption Maisie made her acting debut in Game of Thrones ten years ago

The 22-year-old says she used to find it "impossible" to ignore what people were saying about her on social media.

"It got to me a lot, because there's just a constant feed in your back pocket of what people think of you.

"It gets to a point where you're almost craving something negative so you can sit in a hole of sadness, and it's really bizarre the way it starts to consume you."

Maisie says, eventually, "I just took a step away from it all".
    """

    mentions = me.extract_mentions(text)
    mention_groups = me.group_mentions(mentions)

    print(json.dumps(mention_groups, indent=4))

    predictions, attention_mapping = me.predict_group(mention_groups[0])

    print(json.dumps(predictions, indent=4))
