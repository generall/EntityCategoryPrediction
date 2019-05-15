import torch
from allennlp.common import Registrable
from allennlp.modules import Attention, FeedForward
from allennlp.nn import Activation
from allennlp.nn.util import clone, weighted_sum
from torch.nn import LayerNorm

from category_prediction.model.multi_head_attention import MultiHeadAttention


class SeqCombiner(Registrable):
    """
    This model should take multiple encoded sequences as input and provide single vector output
    """

    def get_output_dim(self):
        raise NotImplementedError()


@SeqCombiner.register("simple-combiner")
class SimpleSeqCombiner(SeqCombiner):
    def get_output_dim(self):
        return self.final_vector_size

    def __init__(self, final_vector_size: int):
        super(SimpleSeqCombiner, self).__init__()
        self.final_vector_size = final_vector_size

    def __call__(
            self,
            encoded_sequences: torch.Tensor,
            encoded_sequences_mask: torch.LongTensor or None,
            encoded_final_states: torch.Tensor
    ):
        """
        :param encoded_sequences: [batch_size * sample_size * seq_length * encoded_seq_dim]
        :param encoded_sequences_mask: [batch_size * sample_size * seq_length]
        :param encoded_final_states: [batch_size * sample_size * input_state_dim]
        :return: [batch_size * output_dim], empty array
        """
        encoded_final_states_max, _ = encoded_final_states.max(dim=1)
        return encoded_final_states_max, []


@SeqCombiner.register('attention-combiner')
class LstmAttentionCombiner(torch.nn.Module, SeqCombiner):
    def get_output_dim(self):
        return self.output_dim

    def __init__(
            self,
            num_layers: int,
            input_state_dim: int,
            encoded_seq_dim: int,
            feed_forward_hidden_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_heads: int,
            dropout_prob: float = 0.1,
            attention_dropout_prob: float = 0.1,
            return_weights: bool = False
    ):
        super(LstmAttentionCombiner, self).__init__()
        self.return_weights = return_weights
        self.num_layers = num_layers
        self.output_dim = output_dim

        self._feed_forwards = []
        self._key_mappers = []
        self._feed_forward_layer_norm_layers = []
        self._attentions = []
        self._layer_norm_layers = []
        self._lstms = []
        self._lstm_ff = []
        self._lstm_norm = []

        self.lstm_input_dim = hidden_dim + encoded_seq_dim  # Encoded state + attention

        feed_forward_input_dim = input_state_dim
        for i in range(num_layers):
            feed_forward = FeedForward(feed_forward_input_dim,
                                       activations=[Activation.by_name('leaky_relu')(),
                                                    Activation.by_name('linear')()],
                                       hidden_dims=[feed_forward_hidden_dim, hidden_dim],
                                       num_layers=2,
                                       dropout=dropout_prob)
            self.add_module(f"feed_forward_layer_{i}", feed_forward)
            self._feed_forwards.append(feed_forward)
            feed_forward_input_dim = hidden_dim

            key_mapper = FeedForward(
                hidden_dim,
                num_layers=1,
                hidden_dims=encoded_seq_dim,
                activations=Activation.by_name('linear')()
            )
            self._key_mappers.append(key_mapper)
            self.add_module(f"key_mapper_layer_{i}", key_mapper)

            feed_forward_layer_norm = LayerNorm(hidden_dim)
            self.add_module(f"feed_forward_layer_norm_{i}", feed_forward_layer_norm)
            self._feed_forward_layer_norm_layers.append(feed_forward_layer_norm)

            attention = MultiHeadAttention(num_heads, attention_dropout_prob)
            self.add_module(f"attention_layer_{i}", attention)
            self._attentions.append(attention)

            layer_norm = LayerNorm(encoded_seq_dim)
            self.add_module(f"layer_norm_{i}", layer_norm)
            self._layer_norm_layers.append(layer_norm)

            lstm = torch.nn.LSTM(
                input_size=self.lstm_input_dim,
                hidden_size=int(hidden_dim / 2),
                batch_first=True,
                bidirectional=True
            )
            self.add_module(f"lstm_{i}", lstm)
            self._lstms.append(lstm)

            lstm_ff = FeedForward(
                hidden_dim,
                num_layers=1,
                hidden_dims=hidden_dim,
                activations=Activation.by_name('linear')()
            )
            self.add_module(f"lstm_ff_{i}", lstm_ff)
            self._lstm_ff.append(lstm_ff)

            lstm_norm = LayerNorm(hidden_dim)
            self.add_module(f"lstm_norm_{i}", lstm_norm)
            self._lstm_norm.append(lstm_norm)

        self._output_projection = FeedForward(
            hidden_dim,
            num_layers=1,
            hidden_dims=output_dim,
            activations=Activation.by_name('leaky_relu')(),
            dropout=dropout_prob
        )

    def forward(
            self,
            encoded_sequences: torch.Tensor,
            encoded_sequences_mask: torch.LongTensor or None,
            encoded_final_states: torch.Tensor
    ):
        """
        Apply LSTM over seq final states with attention over original sequences

        :param encoded_sequences: [batch_size * sample_size * seq_length * encoded_seq_dim]
        :param encoded_sequences_mask: [batch_size * sample_size * seq_length]
        :param encoded_final_states: [batch_size * sample_size * input_state_dim]
        :return:
            output vectors: [batch_size * output_dim]
            attention weights: [batch_size x num_queries x num_heads x seq_length] x num_layers
        """

        batch_size, sample_size, seq_length, encoded_seq_dim = encoded_sequences.shape
        _, _, input_state_dim = encoded_final_states.shape

        flat_size = batch_size * sample_size  # total amount of sentences in all batches

        encoded_sequences = encoded_sequences.view(flat_size, seq_length, -1)
        encoded_final_states = encoded_final_states.view(flat_size, -1)
        encoded_sequences_mask = encoded_sequences_mask.view(flat_size, seq_length)

        store_weights = []

        for i in range(self.num_layers):
            ff: FeedForward = self._feed_forwards[i]
            key_mapper: FeedForward = self._key_mappers[i]
            ff_norm: LayerNorm = self._feed_forward_layer_norm_layers[i]
            att: MultiHeadAttention = self._attentions[i]
            att_norm = self._layer_norm_layers[i]
            lstm: torch.nn.LSTM = self._lstms[i]

            lstm_ff: FeedForward = self._lstm_ff[i]
            lstm_norm: LayerNorm = self._lstm_norm[i]

            # shape: (flat_size * hidden_dim)
            cached_input = encoded_final_states

            # shape: (flat_size * hidden_dim)
            encoded_final_states = ff(encoded_final_states)

            if encoded_final_states.size() == cached_input.size():
                # shape: (flat_size * hidden_dim)
                encoded_final_states = ff_norm(encoded_final_states + cached_input)

            # shape: (flat_size * 1 * encoded_seq_dim)
            encoded_queries = key_mapper(encoded_final_states).unsqueeze(1)

            # attention_lookup shape: (flat_size * 1 * encoded_seq_dim)
            attention_lookup, weights = att(
                queries=encoded_queries,
                keys=encoded_sequences,
                values=encoded_sequences,
                mask=encoded_sequences_mask
            )

            if self.return_weights:
                store_weights.append(weights.cpu().numpy())

            # shape: (flat_size * encoded_seq_dim)
            attention_lookup = att_norm(attention_lookup.squeeze(1))

            # shape: (batch_size * sample_size * (hidden_dim + encoded_seq_dim))
            lstm_input = torch.cat([encoded_final_states, attention_lookup], dim=1).view(batch_size, sample_size, -1)

            # lstm_processed shape: (batch_size, * seq_length * hidden_dim)
            # lstm_hidden shape: (2 * batch_size * hidden_dim)
            lstm_processed, (lstm_hidden, _) = lstm(lstm_input)

            # shape: (flat_size * hidden_dim)
            encoded_final_states = lstm_processed.contiguous().view(batch_size * sample_size, -1)

            if encoded_final_states.size() == cached_input.size():
                encoded_final_states = lstm_norm(lstm_ff(encoded_final_states) + cached_input)

        # shape: (batch_size * hidden_dim)
        final_vectors = lstm_hidden.transpose(0, 1).contiguous().view(batch_size, -1)

        return self._output_projection(final_vectors), store_weights
