from typing import Optional

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics import Auc

from category_prediction.metrics import MulilabelMetric, Metric


class FixedAuc(Auc):
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A one-dimensional tensor of prediction scores of shape (batch_size).
        gold_labels : ``torch.Tensor``, required.
            A one-dimensional label tensor of shape (batch_size), with {1, 0}
            entries for positive and negative class. If it's not binary,
            `positive_label` should be passed in the initialization.
        mask: ``torch.Tensor``, optional (default = None).
            A one-dimensional label tensor of shape (batch_size).
        """

        predictions, gold_labels = self.unwrap_to_tensors(predictions, gold_labels)

        # Sanity checks.
        if gold_labels.dim() != 1:
            raise ConfigurationError("gold_labels must be one-dimensional, "
                                     "but found tensor of shape: {}".format(gold_labels.size()))
        if predictions.dim() != 1:
            raise ConfigurationError("predictions must be one-dimensional, "
                                     "but found tensor of shape: {}".format(predictions.size()))

        # if mask is None:
        #     batch_size = gold_labels.shape[0]
        #     mask = torch.ones(batch_size)
        # mask = mask.byte()
        #
        # self._all_predictions = torch.cat([self._all_predictions,
        #                                    torch.masked_select(predictions, mask).float()], dim=0)
        # self._all_gold_labels = torch.cat([self._all_gold_labels,
        #                                    torch.masked_select(gold_labels, mask).long()], dim=0)

        self._all_predictions = torch.cat([self._all_predictions, predictions], dim=0)
        self._all_gold_labels = torch.cat([self._all_gold_labels, gold_labels], dim=0)


@Metric.register("multilabel-auc")
class MultilabelAuc(MulilabelMetric):
    def __init__(self, labels_count):
        super().__init__(labels_count, FixedAuc)
