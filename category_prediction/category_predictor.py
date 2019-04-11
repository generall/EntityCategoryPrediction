from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn.util import get_mask_from_sequence_lengths, get_text_field_mask, get_lengths_from_binary_sequence_mask, \
    sort_batch_by_length
from torch.nn import Linear


@Model.register("category_predictor")
class CategoryPredictor(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            text_embedder: TextFieldEmbedder,
    ):
        super(CategoryPredictor, self).__init__(vocab)

        self.text_embedder = text_embedder

        self._output_projection_layer = Linear(self.text_embedder.get_output_dim(), vocab.get_vocab_size("labels"))

        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, sentences, categories) -> Dict[str, torch.Tensor]:
        # [batch size * sentence count * sentence length]
        mask = get_text_field_mask(sentences, num_wrapping_dims=1)

        batch_size, sample_size, seq_length = mask.shape

        flat_mask = mask.view(batch_size * sample_size, seq_length)

        lengths = get_lengths_from_binary_sequence_mask(flat_mask)

        sorted_mask, sorted_lengths, restoration_indices, permutation_index = sort_batch_by_length(flat_mask, lengths)

        # [batch x sent.length x embedding]
        embedded = self.text_embedder(sentences).view(batch_size * sample_size, seq_length, -1)[permutation_index]

        sentences_embedding = embedded.sum(dim=1)[restoration_indices]

        sentences_embedding = sentences_embedding.view(batch_size, sample_size, -1)

        sentences_embedding_max, _ = sentences_embedding.max(dim=1)

        outputs = self._output_projection_layer(sentences_embedding_max)

        return {
            'loss': self.loss(outputs, categories.float())
        }
