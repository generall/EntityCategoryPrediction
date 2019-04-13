from collections import defaultdict
from typing import Generic, Type, List, Optional, Union, Tuple, Dict, Callable

import numpy as np
import torch
from allennlp.training.metrics import Metric


class MulilabelMetric(Metric):

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()

    def __init__(
            self,
            labels_count,
            custom_metric_factory: Callable[[], Metric]
    ):
        self.metrics: List[Metric] = [custom_metric_factory() for _ in range(labels_count)]

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Updates metrics for each label separately

        :param predictions: [batch_size * labels_count]
        :param gold_labels: [batch_size * labels_count]
        :param mask: [batch_size * labels_count]
        :return:
        """

        masks = mask.transpose(0, 1) if mask else [None] * len(self.metrics)

        iterator = zip(
            self.metrics,
            predictions.transpose(0, 1),
            gold_labels.transpose(0, 1),
            masks
        )

        for metric, predictions_column, gold_labels_column, mask_column in iterator:
            metric(predictions_column, gold_labels_column, mask_column)

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float]]:
        """
        Average all metrics

        :param reset:
        :return:
        """
        values = [m.get_metric(reset) for m in self.metrics]

        if isinstance(values[0], float):
            return float(np.mean(values))
        elif isinstance(values[0], tuple):
            return tuple(np.array(values).mean(1))
        elif isinstance(values[0], dict):
            sum_dict = defaultdict(float)
            for value in values:
                for key, val in value.items():
                    sum_dict[key] += val

            avg_dict = {}
            for key, val in sum_dict:
                avg_dict[key] = val / len(self.metrics)
            return avg_dict
