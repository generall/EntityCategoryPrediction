from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states
from allennlp.training.metrics import Metric
from torch.nn import Linear

from category_prediction.metrics.multilabel_f1 import MultiLabelF1Measure
from category_prediction.seq_combiner import SeqCombiner


@Model.register("category_predictor")
class CategoryPredictor(Model):

    def __init__(
            self,
            vocab: Vocabulary,
            text_embedder: TextFieldEmbedder,
            encoder: Seq2SeqEncoder,
            seq_combiner: SeqCombiner
    ):
        super(CategoryPredictor, self).__init__(vocab)

        self.text_embedder = text_embedder
        self.encoder = encoder
        self.seq_combiner = seq_combiner

        self._output_projection_layer = Linear(self.seq_combiner.get_output_dim(), vocab.get_vocab_size("labels"))

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.metrics: Dict[str, Metric] = {
            # 'auc': Auc(),
            # "m-auc": MultilabelAuc(vocab.get_vocab_size("labels")),
            'f1': MultiLabelF1Measure()
        }

    def forward(
            self,
            sentences: Dict[str, torch.LongTensor],
            categories: torch.LongTensor = None
    ) -> Dict[str, torch.Tensor]:
        """

        :param sentences: Tensor of word indexes [batch_size * sample_size * seq_length]
        :param categories:
        :return:
        """

        # exclude tensors which are larger then real amount of tokens
        # such as tensors ngram-tensors
        maskable_sentences = dict((key, val) for key, val in sentences.items() if '-ngram' not in key)

        # shape: (batch_size * sample_size * seq_length)
        mask = get_text_field_mask(maskable_sentences, num_wrapping_dims=1)

        batch_size, sample_size, seq_length = mask.shape

        flat_mask = mask.view(batch_size * sample_size, seq_length)

        # lengths = get_lengths_from_binary_sequence_mask(flat_mask)
        # sorted_mask, sorted_lengths, restoration_indices, permutation_index = sort_batch_by_length(flat_mask, lengths)

        # shape: ((batch_size * sample_size) * seq_length * embedding)
        embedded = self.text_embedder(sentences).view(batch_size * sample_size, seq_length, -1)

        # shape: ((batch_size * sample_size) * seq_length * encoder_dim)
        encoder_outputs = self.encoder(embedded, flat_mask)

        # shape: ((batch_size * sample_size), encoder_output_dim)
        final_encoder_output = get_final_encoder_states(
            encoder_outputs,
            flat_mask,
            self.encoder.is_bidirectional()
        )

        # shape: (batch_size * sample_size * encoder_output_dim)
        sentences_embedding = final_encoder_output.view(batch_size, sample_size, -1)

        # shape: (batch_size, sample_size, seq_length, encoder_dim)
        encoder_outputs = encoder_outputs.view(batch_size, sample_size, seq_length, -1)

        mentions_embeddings = self.seq_combiner(encoder_outputs, mask, sentences_embedding)

        outputs = self._output_projection_layer(mentions_embeddings)

        result = {
            'predictions': torch.sigmoid(outputs)
        }

        if categories is not None:
            result['loss'] = self.loss(outputs, categories.float())
            # self.metrics['auc'](outputs.view(-1), categories.view(-1))
            # self.metrics['m-auc'](outputs, categories)
            self.metrics['f1']((outputs > 0.5).long(), categories)

        return result

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return dict((key, metric.get_metric(reset)) for key, metric in self.metrics.items())
