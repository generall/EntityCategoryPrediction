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
                categoty_tag, left_sent, mention, right_sent = line.strip(' ').split('\t')
                sent = f"{left_sent.strip()} {self.left_tag} {mention.strip()} {self.right_tag} {right_sent.strip()}"
                yield sent, categoty_tag

    def _read(self, file_path: str) -> Iterable[Instance]:
        for category_tag, sentences in groupby(self._read_lines(file_path), key=lambda x: x[1]):
            sentences = [s[0] for s in sentences]
            sentences = random.choices(sentences, k=self.sentence_sample)
            yield self.text_to_instance(sentences, category_tag)

    def text_to_instance(self, sentences: List[str], category_tag: str) -> Instance:
        categories = self.category_mapping.get(category_tag)

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
            category_mapping_file: str,
            token_indexers: Dict[str, TokenIndexer],
            tokenizer: Tokenizer = None,
            sentence_sample: int = 5,
            left_tag: str = '@@mb@@',
            right_tag: str = '@@me@@'
    ):
        super().__init__(lazy=True)
        self.category_mapping_file = category_mapping_file
        with open(category_mapping_file) as fd:
            self.category_mapping = json.load(fd)

        self.right_tag = right_tag
        self.left_tag = left_tag
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.sentence_sample = sentence_sample

