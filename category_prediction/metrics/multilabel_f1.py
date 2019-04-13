"""
Found here https://github.com/joelgrus/kaggle-toxic-allennlp/blob/master/toxic/training/metrics/multilabel_f1.py
"""


import torch
from allennlp.training.metrics import Metric


@Metric.register("multilabel-f1")
class MultiLabelF1Measure(Metric):
    """
    Computes multilabel F1. Assumes that predictions are 0 or 1.
    """

    def __init__(self) -> None:
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0

    def __call__(
            self,
            predictions: torch.LongTensor,
            gold_labels: torch.LongTensor,
            mask=None
    ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of 0 and 1 predictions of shape (batch_size, ..., num_labels).
        gold_labels : ``torch.Tensor``, required.
            A tensor of 0 and 1 predictions of shape (batch_size, ..., num_labels).
        """
        self._true_positives += (predictions * gold_labels).sum().item()
        self._false_positives += (predictions * (1 - gold_labels)).sum().item()
        self._true_negatives += ((1 - predictions) * (1 - gold_labels)).sum().item()
        self._false_negatives += ((1 - predictions) * gold_labels).sum().item()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precision : float
        recall : float
        f1-measure : float
        """
        predicted_positives = self._true_positives + self._false_positives
        actual_positives = self._true_positives + self._false_negatives

        precision = self._true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = self._true_positives / actual_positives if actual_positives > 0 else 0

        if precision + recall > 0:
            f1_measure = 2 * precision * recall / (precision + recall)
        else:
            f1_measure = 0

        if reset:
            self.reset()
        return f1_measure

    def reset(self):
        self._true_positives = 0.0
        self._true_negatives = 0.0
        self._false_positives = 0.0
        self._false_negatives = 0.0
