from typing import Dict

import torch
from overrides import overrides
from allennlp.data import TokenIndexer, Tokenizer, DataArray, Vocabulary
from allennlp.data.fields import TextField, SequenceField


class LazyTextFiled(SequenceField[Dict[str, torch.Tensor]]):
    """
    Same as ``TextField``, but performs tokenization only if needed.
    Required for Multiprocessing indexer
    """

    tokenizers = {}

    def __init__(
            self,
            text: str,
            tokenizer_name: str,
            token_indexers: Dict[str, TokenIndexer]
    ):
        self.token_indexers = token_indexers
        self.text = text

        self.tokenizer_name = tokenizer_name

        self.text_field = None

        self.padding_lengths = None

    def _get_filed(self):
        if self.text_field is None:
            if self.tokenizer_name not in self.tokenizers:
                print(self.tokenizers)
                raise RuntimeError(f"No tokenizer {self.tokenizer_name}")

            tokenized_text = self.tokenizers[self.tokenizer_name].tokenize(self.text)
            self.text_field = TextField(tokenized_text, self.token_indexers)
        return self.text_field

    @overrides
    def count_vocab_items(self, *args, **kwargs):
        return self._get_filed().count_vocab_items(*args, **kwargs)

    @overrides
    def sequence_length(self, *args, **kwargs) -> int:
        return self._get_filed().sequence_length(*args, **kwargs)

    @overrides
    def empty_field(self, *args, **kwargs) -> 'SequenceField':
        return self._get_filed().empty_field(*args, **kwargs)

    @overrides
    def get_padding_lengths(self, *args, **kwargs) -> Dict[str, int]:
        if self.padding_lengths is None:
            self.padding_lengths = self._get_filed().get_padding_lengths(*args, **kwargs)
        return self.padding_lengths

    @overrides
    def as_tensor(self, *args, **kwargs) -> DataArray:
        return self._get_filed().as_tensor(*args, **kwargs)

    @overrides
    def index(self, *args, **kwargs):
        return self._get_filed().index(*args, **kwargs)

    @overrides
    def batch_tensors(self, *args, **kwargs):
        return self._get_filed().batch_tensors(*args, **kwargs)

