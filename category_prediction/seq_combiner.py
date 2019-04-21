import torch
from allennlp.common import Registrable
from allennlp.modules import Attention, FeedForward
from allennlp.nn import Activation
from allennlp.nn.util import clone, weighted_sum


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
        :return: [batch_size * output_dim]
        """
        encoded_final_states_max, _ = encoded_final_states.max(dim=1)
        return encoded_final_states_max


class AttentionLSTM(torch.nn.Module):
    def __init__(
            self,
            num_layers,
            input_state_dim,
            encoded_seq_dim,
            output_dim,
            dropout: float,
            attention: Attention
    ):
        super(AttentionLSTM, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.attentions = clone(attention, num_layers)
        for att in self.attentions:
            att.reset_parameters()

        self.lstm_input_dim = input_state_dim + encoded_seq_dim  # Encoded state + attention

        self.lstms = [
            torch.nn.LSTM(
                input_size=self.lstm_input_dim,
                hidden_size=int(input_state_dim / 2),
                batch_first=True,
                bidirectional=True
            )
            for _ in range(num_layers)
        ]

        self._output_projection = FeedForward(
            input_state_dim,
            num_layers=1,
            hidden_dims=output_dim,
            activations=Activation.by_name('leaky_relu')(),
            dropout=dropout
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
        :return: [batch_size * output_dim]
        """

        batch_size, sample_size, seq_length, encoded_seq_dim = encoded_sequences.shape
        _, _, input_state_dim = encoded_final_states.shape

        flat_size = batch_size * sample_size  # total amount of sentences in all batches

        encoded_sequences = encoded_sequences.view(flat_size, seq_length, -1)
        encoded_final_states = encoded_final_states.view(flat_size, -1)
        encoded_sequences_mask = encoded_sequences_mask.view(flat_size, seq_length)

        for i in range(self.num_layers):
            att: Attention = self.attentions[i]

            lstm: torch.nn.LSTM = self.lstms[i]

            # shape: (flat_size * seq_length)
            weights = att(encoded_final_states, encoded_sequences, encoded_sequences_mask)

            # shape: (flat_size * encoded_seq_dim)
            attention_lookup = weighted_sum(encoded_sequences, weights)

            # shape: (batch_size * sample_size * (input_state_dim + encoded_seq_dim))
            lstm_input = torch.cat([encoded_final_states, attention_lookup], dim=1).view(batch_size, sample_size, -1)

            # lstm_processed shape: (batch_size, * seq_length * input_state_dim)
            # lstm_hidden shape: (2 * batch_size * input_state_dim)
            lstm_processed, (lstm_hidden, _) = lstm(lstm_input)

            # shape: (flat_size * input_state_dim)
            encoded_final_states = lstm_processed.contiguous().view(batch_size * sample_size, -1)

        # shape: (batch_size * input_state_dim)
        final_vectors = lstm_hidden.transpose(0, 1).contiguous().view(batch_size, input_state_dim)

        return self._output_projection(final_vectors)


@SeqCombiner.register('attention-combiner')
class LstmAttentionCombiner(SeqCombiner, AttentionLSTM):
    def get_output_dim(self):
        return self.output_dim
