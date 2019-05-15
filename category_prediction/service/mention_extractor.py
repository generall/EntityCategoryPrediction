import json
import os
import re
from pprint import pprint
from typing import List

import spacy
from allennlp.models import load_archive
from allennlp.predictors import Predictor

from category_prediction.predictor import PersonPredictor
from category_prediction.settings import DATA_DIR


class MentionExtractor:

    def __init__(self, model_path, context_size=100, label='PERSON'):
        self.label = label
        self.context_size = context_size
        self.ner = spacy.load("en_core_web_sm")
        archive = load_archive(model_path, overrides=PersonPredictor.overrides)
        self.predictor = Predictor.from_archive(archive, 'person-predictor')

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
