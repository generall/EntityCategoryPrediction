import torch
from allennlp.common import Registrable


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

    def __init__(self, final_vector_size):
        self.final_vector_size = final_vector_size

    def __call__(self, encoded_sequences, encoded_final_states):
        """
        :param encoded_sequences: [batch_size * sample_size * embedding_size]
        :param encoded_final_states: [batch_size * embedding_size]
        :return: [batch_size * final_vector_size]
        """
        encoded_final_states_max, _ = encoded_final_states.max(dim=1)
        return encoded_final_states_max
