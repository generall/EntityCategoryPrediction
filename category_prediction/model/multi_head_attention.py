import torch
from allennlp.nn.util import masked_softmax, weighted_sum
from torch.nn import Linear, Dropout

from allennlp.common import Registrable


class MultiHeadAttention(torch.nn.Module, Registrable):

    def __init__(
            self,
            num_heads: int,
            attention_dropout_prob: float = 0.1
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self._attention_dropout = Dropout(attention_dropout_prob)

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            mask: torch.Tensor = None
    ):
        """

        :param queries: [batch_size x num_queries x query_size]
        :param keys: [batch_size x seq_length x query_size]
        :param values: [batch_size x seq_length x value_size]
        :param mask: [batch_size x seq_length]

        :return:
            outputs: [batch_size x num_queries x value_size]
            attention: [batch_size x num_queries x num_heads x seq_length]
        """

        num_heads = self.num_heads
        queries_batch_size, num_queries, query_size = queries.shape
        value_batch_size, val_seq_length, value_size = values.shape
        keys_batch_size, keys_seq_length, keys_size = keys.shape

        assert queries_batch_size == value_batch_size
        assert keys_batch_size == value_batch_size
        assert keys_batch_size == value_batch_size
        assert val_seq_length == keys_seq_length

        assert query_size == keys_size

        seq_length = val_seq_length
        batch_size = queries_batch_size

        scale = (query_size // num_heads) ** 0.5

        # Shape (num_heads * batch_size, num_queries, attention_dim / num_heads)
        queries_per_head = queries.view(batch_size, num_queries, num_heads, query_size // num_heads)
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * num_heads, num_queries, query_size // num_heads)

        if queries.data_ptr() == values.data_ptr():
            values_per_head = queries_per_head
        else:
            # Shape (num_heads * batch_size, seq_length, value_size / num_heads)
            values_per_head = values.view(batch_size, seq_length, num_heads, value_size // num_heads)
            values_per_head = values_per_head.transpose(1, 2).contiguous()
            values_per_head = values_per_head.view(batch_size * num_heads, seq_length, value_size // num_heads)

        if values.data_ptr() == keys.data_ptr():
            keys_per_head = values_per_head
        else:
            # Shape (num_heads * batch_size, seq_length, attention_dim / num_heads)
            keys_per_head = keys.view(batch_size, seq_length, num_heads, query_size // num_heads)
            keys_per_head = keys_per_head.transpose(1, 2).contiguous()
            keys_per_head = keys_per_head.view(batch_size * num_heads, seq_length, query_size // num_heads)

        # Shape (num_heads * batch_size, attention_dim / num_heads, seq_length)
        keys_per_head = keys_per_head.transpose(1, 2)

        # shape (num_heads * batch_size, num_queries, seq_length)
        scaled_similarities = torch.bmm(queries_per_head / scale, keys_per_head)

        # shape (num_heads * batch_size, num_queries, seq_length)
        # Normalise the distributions, using the same mask for all heads.
        attention = masked_softmax(scaled_similarities,
                                   mask.repeat(1, num_heads).view(batch_size * num_heads, seq_length),
                                   memory_efficient=False)

        attention = self._attention_dropout(attention)

        # Take a weighted sum of the values with respect to the attention
        # distributions for each element in the num_heads * batch_size dimension.
        # shape (num_heads * batch_size, num_queries, value_size/num_heads)
        outputs = weighted_sum(values_per_head, attention)

        # Reshape back to original shape (batch_size, seq_length, value_size)
        # shape (batch_size, num_heads, num_queries, value_size/num_heads)
        outputs = outputs.view(batch_size, num_heads, num_queries, value_size // num_heads)
        # shape (batch_size, num_queries, num_heads, value_size/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, num_queries, value_size)
        outputs = outputs.view(batch_size, num_queries, value_size)

        attention = attention.view(batch_size, num_heads, num_queries, seq_length).transpose(1, 2)

        return outputs, attention


if __name__ == '__main__':
    matt = MultiHeadAttention(num_heads=2)

    queries = torch.rand(3, 5, 22)
    keys = torch.rand(3, 7, 22)
    values = torch.rand(3, 7, 26)
    mask = torch.rand(3, 7) > 0.5

    res, weights = matt.forward(queries, keys, values, mask)

    print("res.shape", res.shape)
    print("weights.shape", weights.shape)

    print(mask)
    print(weights)

