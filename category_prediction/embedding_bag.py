import json
import os
from functools import partial

import gensim
import numpy as np
import torch
from allennlp.data import Vocabulary
from allennlp.modules import TokenEmbedder
from gensim.models.fasttext import FastText
from gensim.models.utils_any2vec import ft_ngram_hashes
from torch.nn import EmbeddingBag


def load_fasttext_model(path):
    try:
        model = FastText.load(path).wv
    except Exception as e:
        try:
            model = FastText.load_fasttext_format(path).wv
        except Exception as e:
            model = gensim.models.KeyedVectors.load(path)

    return model


class FastTextEmbeddingBag(EmbeddingBag):

    @classmethod
    def get_params_path(cls, model_path):
        return model_path + '.params'

    def __init__(
            self,
            model_path: str,
            model_params_path: str,
            trainable=False
    ):
        self.learn_emb = trainable
        self.model_path = model_path
        self.model_params_path = model_params_path or self.get_params_path(model_path)
        self.hash_params = {}
        self.vocab = {}
        self.dimension = 0
        self.num_vectors = 0

        weights = None

        if self.model_path and os.path.exists(self.model_path):
            ft = self.load_ft_model(self.model_path)
            weights = np.concatenate([ft.vectors_vocab, ft.vectors_ngrams], axis=0)
        else:
            if os.path.exists(self.model_params_path):
                # Assume weights will be loaded later
                self.load_saved_params(self.model_params_path)
            else:
                raise RuntimeError("Neither model file no param file found")

        super().__init__(self.num_vectors, self.dimension)

        if weights is not None:
            self.weight.data.copy_(torch.FloatTensor(weights))

        self.weight.requires_grad = trainable

    def load_saved_params(self, model_param_path):
        with open(model_param_path, encoding="utf-8") as fd:
            ft_params = json.load(fd)
            self.hash_params = ft_params['hash_params']
            self.vocab = ft_params['vocab']
            self.dimension = ft_params['dimension']
            self.num_vectors = ft_params['num_vectors']

    def load_ft_model(self, model_path):

        self.model_params_path = self.get_params_path(model_path)
        try:
            ft = FastText.load(model_path).wv
        except Exception as e:
            ft = FastText.load_fasttext_format(model_path).wv

        self.hash_params = {
            "minn": ft.min_n,
            "maxn": ft.max_n,
            "num_buckets": ft.bucket,
            "fb_compatible": ft.compatible_hash,
        }
        self.dimension = ft.vector_size
        self.num_vectors = ft.vectors_vocab.shape[0] + ft.vectors_ngrams.shape[0]

        self.vocab = dict((word, keydvector.index) for word, keydvector in ft.vocab.items())

        with open(self.model_params_path, 'w', encoding="utf-8") as out:
            json.dump({
                'hash_params': self.hash_params,
                'vocab': self.vocab,
            }, out, ensure_ascii=False, indent=2)

        return ft

    def get_subword_ids(self, word):
        if word in self.vocab:
            return [self.vocab[word]]
        res = []
        for ngram_id in ft_ngram_hashes(word, **self.hash_params):
            res.append(ngram_id + len(self.vocab))

        return res

    def forward(self, words, offsets=None):
        word_subinds = []
        word_offsets = [0]
        for word in words:
            subinds = self.get_subword_ids(word)
            word_subinds += subinds
            word_offsets.append(word_offsets[-1] + len(subinds))
        word_offsets = word_offsets[:-1]

        ind = torch.LongTensor(word_subinds)
        offsets = torch.LongTensor(word_offsets)

        if self.weight.is_cuda:
            ind = ind.cuda()
            offsets = offsets.cuda()

        return super().forward(ind, offsets)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(FastTextEmbeddingBag, self).state_dict(destination, prefix, keep_vars)

        if not self.learn_emb:
            state_dict.pop(prefix + 'weight')

        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        strict = self.learn_emb  # restore weights only in case they are trainable
        return super(FastTextEmbeddingBag, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                                       missing_keys, unexpected_keys, error_msgs)


class GensimFasttext:
    def __init__(
            self,
            model_path: str
    ):
        self.model = load_fasttext_model(model_path)

    def __call__(self, words):
        vectors = np.stack([self.model.get_vector(word) for word in words]).astype(np.float32)
        return torch.from_numpy(vectors)

    @property
    def dimension(self):
        return self.model.wv.vector_size


@TokenEmbedder.register("fasttext-embedder")
class FasttextTokenEmbedder(TokenEmbedder):

    def __init__(
            self,
            vocab: Vocabulary,
            model_path: str,
            model_params_path: str,
            trainable: bool = False,
            vocab_namespace: str = "tokens",
            force_cpu: bool = False
    ):
        super(FasttextTokenEmbedder, self).__init__()
        self.vocab_namespace = vocab_namespace
        self.vocab = vocab

        self.projection_layer = torch.nn.Linear(1, 1)  # is not used yet

        if force_cpu:
            assert not trainable, "Can't train weight with force_cpu mode"
            self.embedding = GensimFasttext(model_path=model_path)
        else:
            self.embedding = FastTextEmbeddingBag(
                model_path=model_path,
                model_params_path=model_params_path,
                trainable=trainable
            )

    def get_output_dim(self) -> int:
        return self.embedding.dimension

    def forward(self, inputs: torch.Tensor):

        original_size = inputs.size()
        inputs = inputs.view(-1)

        get_token_index = partial(self.vocab.get_token_from_index, namespace=self.vocab_namespace)
        words = map(get_token_index, inputs.tolist())

        embedded = self.embedding(words)

        view_args = list(original_size) + [embedded.size(-1)]
        embedded = embedded.view(*view_args)

        if inputs.is_cuda:
            embedded = embedded.cuda()

        return embedded
