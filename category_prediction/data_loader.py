import json
import random
from itertools import groupby
from typing import Dict, Iterable, List

from allennlp.common.util import pad_sequence_to_length
from allennlp.data import DatasetReader, TokenIndexer, Tokenizer, Instance, TokenType, Token, Vocabulary, DataIterator
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField, MultiLabelField


@DatasetReader.register("mention_categories")
class MenionsLoader(DatasetReader):

    def _read_lines(self, file_path: str):

        with open(file_path) as fd:
            for line in fd:
                sent, categoty_tag = line.strip().split('\t')
                yield sent, categoty_tag

    def _read(self, file_path: str) -> Iterable[Instance]:
        for category_tag, sentences in groupby(self._read_lines(file_path), key=lambda x: x[1]):
            sentences = [s[0] for s in sentences]
            sentences = random.choices(sentences, k=self.sentence_sample)
            yield self.text_to_instance(sentences, category_tag)

    def text_to_instance(self, sentences: List[str], category_tag: str) -> Instance:
        categories = extract_categories(category_tag)

        sentence_fields = []
        for sentence in sentences:
            tokenized_sentence = self.tokenizer.tokenize(sentence)
            sent_field = TextField(tokenized_sentence, self.token_indexers)
            sentence_fields.append(sent_field)

        return Instance({
            'sentences': ListField(sentence_fields),
            'categories': MultiLabelField(categories)
        })

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer],
            tokenizer: Tokenizer = None,
            sentence_sample: int = 5
    ):
        super().__init__(lazy=True)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.sentence_sample = sentence_sample


def extract_categories(category_tag):
    return category_tag.split('_')
