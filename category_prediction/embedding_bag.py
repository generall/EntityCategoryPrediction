import json
import os
from functools import partial

import gensim
import numpy as np
import torch
from allennlp.data import Vocabulary
from allennlp.modules import TokenEmbedder
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import Vocab
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

    def get_output_dim(self) -> int:
        return self.dimension

    def __init__(
            self,
            model_path: str,
            trainable=False
    ):
        self.learn_emb = trainable
        self.model_path = model_path

        weights = None

        if self.model_path and os.path.exists(self.model_path):
            ft = load_fasttext_model(self.model_path)

            self.num_vectors = ft.vectors_vocab.shape[0] + ft.vectors_ngrams.shape[0]

            self.dimension = ft.vector_size
            weights = np.concatenate([ft.vectors_vocab, ft.vectors_ngrams], axis=0)
        else:
            raise RuntimeError("Neither model file no param file found")

        super().__init__(self.num_vectors, self.dimension)

        if weights is not None:
            self.weight.data.copy_(torch.FloatTensor(weights))

        self.weight.requires_grad = trainable

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


@TokenEmbedder.register("fasttext-embedder")
class FastTextTokenEmbedder(TokenEmbedder):

    def get_output_dim(self) -> int:
        return self.embedding.get_output_dim()

    def __init__(
            self,
            model_path: str,
            trainable=False
    ):
        super(FastTextTokenEmbedder, self).__init__()
        self.embedding = FastTextEmbeddingBag(model_path, trainable)

    def forward(self, indexes: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor):
        # 1D Tensor
        raw_indexes = torch.masked_select(indexes, mask.byte())

        original_size = lengths.size()

        # Convert ngram lengths into offsets. Offset = start position of each word
        offsets = lengths.view(-1).cumsum(dim=0).roll(1)
        offsets[0] = 0

        # shape: (total_words, embedding_size)
        embedded = self.embedding(raw_indexes, offsets)

        # shape: (batch_size, num_sentences, sent_length, embeddings)
        view_args = list(original_size) + [embedded.size(-1)]

        embedded = embedded.view(*view_args)

        return embedded


class GensimFasttext:
    def __init__(
            self,
            model_path: str
    ):
        self.model = load_fasttext_model(model_path)

    def __call__(self, words):
        vectors = np.stack([self.model.get_vector(word) for word in words]).astype(np.float32)
        return torch.from_numpy(vectors)

    def get_output_dim(self) -> int:
        return self.model.vector_size


@TokenEmbedder.register("gensim-embedder")
class GensimTokenEmbedder(TokenEmbedder):

    def get_output_dim(self) -> int:
        return self.embedding.get_output_dim()

    def __init__(
            self,
            vocab: Vocabulary,
            model_path: str = None,
            vocab_namespace: str = "tokens"
    ):
        self.model_path = model_path
        self.vocab_namespace = vocab_namespace
        self.vocab = vocab

        super(GensimTokenEmbedder, self).__init__()

        # self.projection_layer = torch.nn.Linear(1, 1)  # is not used yet

        if model_path is not None:
            self.embedding = GensimFasttext(model_path=model_path)
        else:
            self.embedding = None

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


class OnDiskFastText:

    def load_matrix(self, file_path):
        st = os.stat(file_path)
        file_size = st.st_size
        embedding_count = (file_size - self.header_size_bites) // self.dtype_size // self.dim
        matrix = np.memmap(file_path, dtype=self.dtype, mode='r', shape=(embedding_count, self.dim),
                           offset=self.header_size_bites)
        return matrix

    def __init__(
            self,
            model_path,
            model_params_path,
            dim,
            dtype="float32",
            dtype_size=4,
            header_size=32
    ):
        self.header_size_bites = header_size * dtype_size
        self.dtype_size = dtype_size
        self.dtype = dtype

        self.dim = dim

        with open(model_params_path) as fd:
            params = json.load(fd)

        self.model = gensim.models.keyedvectors.FastTextKeyedVectors(
            vector_size=dim,
            min_n=params['hash_params']['minn'],
            max_n=params['hash_params']['maxn'],
            bucket=params['hash_params']['num_buckets'],
            compatible_hash=params['hash_params']['fb_compatible']
        )

        self.model.vectors_vocab = self.load_matrix(f'{model_path}.vectors_vocab.npy')
        self.model.vectors = self.load_matrix(f'{model_path}.vectors.npy')
        self.model.vectors_ngrams = self.load_matrix(f'{model_path}.vectors_ngrams.npy')
        self.model.vocab = dict((word, Vocab(index=idx, count=1)) for word, idx in params['vocab'].items())

    def __call__(self, words):
        vectors = np.stack([self.model.get_vector(word) for word in words]).astype(np.float32)
        return torch.from_numpy(vectors)

    def get_output_dim(self) -> int:
        return self.dim


@TokenEmbedder.register("disk-gensim-embedder")
class DiskGensimTokenEmbedder(GensimTokenEmbedder):

    def __init__(
            self,
            vocab: Vocabulary,
            model_path: str,
            model_params_path: str,
            dimensions: int,
            vocab_namespace: str = "tokens",
            trainable: bool = False  # used for compatibility
    ):
        super(DiskGensimTokenEmbedder, self).__init__(vocab, model_path=None, vocab_namespace=vocab_namespace)

        self.model_path = model_path

        self.embedding = OnDiskFastText(
            model_path=model_path,
            model_params_path=model_params_path,
            dim=dimensions
        )
