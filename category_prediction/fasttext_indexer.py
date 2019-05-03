import json
import os
from typing import Dict, List

from allennlp.common.util import pad_sequence_to_length
from allennlp.data import TokenIndexer, TokenType, Token, Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer
from gensim.models.utils_any2vec import ft_ngram_hashes

from category_prediction import load_fasttext_model


@TokenIndexer.register("fasttext-id")
class FasttextTokenIndexer(TokenIndexer[int]):

    def __init__(
            self,
            model_path,
            namespace: str = 'tokens',
            lowercase_tokens: bool = False,
            model_params_path=None
    ):
        self.model_path = model_path
        self.model_params_path = model_params_path or self.get_params_path(model_path)
        self.hash_params = {}
        self.vocab = {}
        self.num_vectors = 0

        self.single_id_indexer = SingleIdTokenIndexer(
            namespace,
            lowercase_tokens
        )  # ToDo: Add start and end tokens params

        if os.path.exists(self.model_params_path):
            # Assume weights will be loaded later
            self.load_saved_params(self.model_params_path)
        else:
            self.load_ft_model(model_path)

    @classmethod
    def get_params_path(cls, model_path):
        return model_path + '.params'

    def load_saved_params(self, model_param_path):
        with open(model_param_path, encoding="utf-8") as fd:
            ft_params = json.load(fd)
            self.hash_params = ft_params['hash_params']
            self.vocab = ft_params['vocab']

    def load_ft_model(self, model_path):
        self.model_params_path = self.get_params_path(model_path)
        ft = load_fasttext_model(model_path)

        self.hash_params = {
            "minn": ft.min_n,
            "maxn": ft.max_n,
            "num_buckets": ft.bucket,
            "fb_compatible": ft.compatible_hash,
        }

        self.vocab = dict((word, keydvector.index) for word, keydvector in ft.vocab.items())

        with open(self.model_params_path, 'w', encoding="utf-8") as out:
            json.dump({
                'hash_params': self.hash_params,
                'vocab': self.vocab,
            }, out, ensure_ascii=False, indent=2)

    def words_to_indexes(self, words):
        words_ngram_ids = []
        word_offsets = [0]
        mask = []
        for word in words:
            ngram_ids = self.get_ngram_ids(word)
            words_ngram_ids += ngram_ids
            mask += [1] * len(ngram_ids)
            word_offsets.append(word_offsets[-1] + len(ngram_ids))

        word_offsets = word_offsets[:-1]

        return words_ngram_ids, word_offsets, mask

    def get_ngram_ids(self, word):
        if word in self.vocab:
            return [self.vocab[word]]
        res = []
        for ngram_id in ft_ngram_hashes(word, **self.hash_params):
            res.append(ngram_id + len(self.vocab))

        return res

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        return self.single_id_indexer.count_vocab_items(token, counter)

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary, index_name: str) -> Dict[
        str, List[TokenType]]:
        words = [token.text for token in tokens]
        word_ngram_ids, word_offsets, mask = self.words_to_indexes(words)

        return {
            f"{index_name}-ngram": word_ngram_ids,
            f"{index_name}-ngram-offsets": word_offsets,
            f"{index_name}-ngram-mask": mask,
            **self.single_id_indexer.tokens_to_indices(tokens, vocabulary, index_name)
        }

    def get_padding_token(self) -> TokenType:
        return 0

    def get_padding_lengths(self, token: TokenType) -> Dict[str, int]:
        return {}

    def pad_token_sequence(self, tokens: Dict[str, List[TokenType]], desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[TokenType]]:
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}
