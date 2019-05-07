import logging
import os
from multiprocessing import Process, Queue
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import DataIterator, Instance
from allennlp.data.iterators import MultiprocessIterator
from allennlp.data.iterators.data_iterator import TensorDict
from allennlp.data.iterators.multiprocess_iterator import _queuer

logger = logging.getLogger(__name__)


@DataIterator.register("numpy-multiprocessing")
class NumpyItearator(MultiprocessIterator):

    @classmethod
    def tensor_to_numpy(cls, obj: Any):
        if isinstance(obj, dict):
            return dict((key, cls.tensor_to_numpy(val)) for key, val in obj.items())
        elif isinstance(obj, list):
            return [cls.tensor_to_numpy(val) for val in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.numpy()
        else:
            return obj

    @classmethod
    def numpy_to_tensor(cls, obj: Any):
        if isinstance(obj, dict):
            return dict((key, cls.numpy_to_tensor(val)) for key, val in obj.items())
        elif isinstance(obj, list):
            return [cls.numpy_to_tensor(val) for val in obj]
        elif isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        else:
            return obj

    @classmethod
    def _queuer(cls, *args, **kwargs):
        # print("Querer pid:", os.getpid())
        return _queuer(*args, **kwargs)

    @classmethod
    def _create_tensor_dicts(cls, input_queue: Queue,
                             output_queue: Queue,
                             iterator: DataIterator,
                             shuffle: bool,
                             index: int) -> None:
        """
        Pulls at most ``max_instances_in_memory`` from the input_queue,
        groups them into batches of size ``batch_size``, converts them
        to ``TensorDict`` s, and puts them on the ``output_queue``.
        """

        # print("Worker pid:", os.getpid())

        def instances() -> Iterator[Instance]:
            instance = input_queue.get()
            while instance is not None:
                yield instance
                instance = input_queue.get()

        for tensor_dict in iterator(instances(), num_epochs=1, shuffle=shuffle):
            numpy_dict = cls.tensor_to_numpy(tensor_dict)
            output_queue.put(numpy_dict)

        output_queue.put(index)

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterator[TensorDict]:

        # If you run it forever, the multiprocesses won't shut down correctly.
        # TODO(joelgrus) find a solution for this
        if num_epochs is None:
            raise ConfigurationError("Multiprocess Iterator must be run for a fixed number of epochs")

        output_queue = Queue(self.output_queue_size)
        input_queue = Queue(self.output_queue_size * self.batch_size)

        # Start process that populates the queue.
        self.queuer = Process(target=self._queuer, args=(instances, input_queue, self.num_workers, num_epochs))
        self.queuer.start()

        logger.info(f"queuer.pid: {self.queuer.pid}")

        # Start the tensor-dict workers.
        for i in range(self.num_workers):
            args = (input_queue, output_queue, self.iterator, shuffle, i)
            process = Process(target=self._create_tensor_dicts, args=args)
            process.start()
            self.processes.append(process)

        def get_shapes(item):
            if isinstance(item, dict):
                return [(key, get_shapes(val)) for key, val in item.items()]
            else:
                return item.shape

        num_finished = 0
        num_items = 0
        while num_finished < self.num_workers:
            item = output_queue.get()
            num_items += 1
            if isinstance(item, int):
                num_finished += 1
                logger.info(f"worker {item} finished ({num_finished} / {self.num_workers})")
            else:
                # shapes = get_shapes(item)
                # if num_items % 5 == 0:
                #     logger.info(
                #         f"item.shape {shapes}, input_queue, {input_queue.qsize()}, out_queue, {output_queue.qsize()}")
                yield self.numpy_to_tensor(item)

        for process in self.processes:
            process.join()
        self.processes.clear()

        if self.queuer is not None:
            self.queuer.join()
            self.queuer = None
