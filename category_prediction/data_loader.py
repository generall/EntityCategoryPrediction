import glob
import json
import logging
import random
import re
from itertools import groupby
from multiprocessing import Manager, Queue, Process, get_logger
from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, TokenIndexer, Tokenizer, Instance
from allennlp.data.fields import TextField, ListField, MultiLabelField

from category_prediction.lazy_text_field import LazyTextFiled

logger = get_logger()  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)


@DatasetReader.register("parallel_reader")
class ParallelLoader(DatasetReader):

    def _file_reader(self, output: Queue, file_path: str):
        items = 0
        for instance in self.reader._read(file_path):
            items += 1
            output.put(instance)

        output.put(items)

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Assume file_path is actually prefix for real data files

        :param file_path:
        :return:
        """
        files = glob.glob(file_path)
        if len(files) == 1:
            yield from self.reader._read(files[0])
        else:
            output_queue = Queue(self.output_queue_size)

            for file in files:
                args = (output_queue, file)
                process = Process(target=self._file_reader, args=args)
                if len(self.running_processes) < self.parallel:
                    process.start()
                    self.running_processes.append(process)
                else:
                    self.waiting_processes.append(process)

            num_finished = 0
            while num_finished < len(files):
                item = output_queue.get()
                if isinstance(item, int):
                    num_finished += 1
                    logger.info(f"loader {item} finished ({num_finished} / {len(files)})")

                    if len(self.waiting_processes) > 0:
                        process = self.waiting_processes.pop()
                        process.start()
                        self.running_processes.append(process)
                else:
                    yield item

            for process in self.running_processes:
                process.join()
            self.running_processes.clear()

    def text_to_instance(self, *inputs) -> Instance:
        return self.reader.text_to_instance(*inputs)

    def __init__(self, reader: DatasetReader, parallel: int = 1, output_queue_size: int = 1000):
        super().__init__(lazy=True)
        self.output_queue_size = output_queue_size
        self.parallel = parallel
        self.reader: DatasetReader = reader

        self.waiting_processes: List[Process] = []
        self.running_processes: List[Process] = []


@DatasetReader.register("mention_categories")
class MenionsLoader(DatasetReader):

    def _read_lines(self, file_path: str):

        with open(file_path, encoding="utf-8") as fd:
            for line in fd:
                category_tag, left_sent, mention, right_sent = line.strip(' ').split('\t')
                # sent = f"{left_sent.strip()} {self.left_tag} {mention.strip()} {self.right_tag} {right_sent.strip()}"
                yield (left_sent.strip(), mention.strip(), right_sent.strip()), category_tag

    def sentence_constructor(self, left_sent, mention, right_sent):
        return f"{left_sent} {self.left_tag} {mention} {self.right_tag} {right_sent}"

    def select_sentences(self, sentences):
        return random.choices(sentences, k=self.sentence_sample)

    def _read(self, file_path: str) -> Iterable[Instance]:
        for category_tag, rows in groupby(self._read_lines(file_path), key=lambda x: x[1]):
            sentences = []
            for row, _ in rows:
                left_sent, mention, right_sent = row
                sentences.append(self.sentence_constructor(left_sent, mention, right_sent))

            sentences = self.select_sentences(sentences)
            yield self.text_to_instance(sentences, category_tag)

    def text_to_instance(self, sentences: List[str], category_tag: str = None) -> Instance:

        sentence_fields = []
        for sentence in sentences:
            sent_field = LazyTextFiled(
                text=sentence,
                tokenizer_name="sentences",
                token_indexers=self.token_indexers
            )
            sentence_fields.append(sent_field)

        dt = {
            'sentences': ListField(sentence_fields),
        }

        if category_tag is not None:
            categories = self.category_mapping.get(category_tag)
            dt['categories'] = MultiLabelField(categories)

        return Instance(dt)

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer],
            category_mapping_file: str = None,
            tokenizer: Tokenizer = None,
            sentence_sample: int = 5,
            left_tag: str = '@@mb@@',
            right_tag: str = '@@me@@'
    ):
        super().__init__(lazy=True)
        self.category_mapping_file = category_mapping_file

        if category_mapping_file is not None:
            with open(category_mapping_file, encoding="utf-8") as fd:
                self.category_mapping = json.load(fd)

        self.right_tag = right_tag
        self.left_tag = left_tag
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.sentence_sample = sentence_sample

        LazyTextFiled.tokenizers["sentences"] = self.tokenizer


@DatasetReader.register("augmented_mention_categories")
class AugmentedMenionsLoader(MenionsLoader):

    skip_mention_prob = 0.5
    shrink_to_sent_prob = 0.25
    shrink_prob = 0.25
    resample_prob = 0.3

    sent_re = re.compile(r"[?!.]")

    def sentence_constructor(self, left_sent, mention, right_sent):

        if random.random() < self.skip_mention_prob:
            mention = ""

        if random.random() < self.shrink_to_sent_prob:
            right_sent = re.split(self.sent_re, right_sent)[0]
            left_sent = re.split(self.sent_re, left_sent)[-1]
        elif random.random() < self.shrink_prob:
            right_sent = right_sent[:random.randint(0, len(right_sent))]
            left_sent = right_sent[random.randint(0, len(left_sent)):]

        return f"{left_sent} {self.left_tag} {mention} {self.right_tag} {right_sent}"

    def select_sentences(self, sentences):
        if random.random() > self.resample_prob:
            resample_size = random.randint(1, 2)
            return random.choices(random.choices(sentences, k=resample_size), k=self.sentence_sample)
        else:
            return random.choices(sentences, k=self.sentence_sample)


@DatasetReader.register("vocab_mention_categories")
class VocabMentionsLoader(MenionsLoader):

    def _read(self, file_path: str) -> Iterable[Instance]:
        for sent, category_tag in self._read_lines(file_path):
            yield self.text_to_instance(sent, category_tag)

    def text_to_instance(self, sentence: str, category_tag: str) -> Instance:
        categories = self.category_mapping.get(category_tag)

        tokenized_sentence = self.tokenizer.tokenize(sentence)
        sent_field = TextField(tokenized_sentence, self.token_indexers)

        return Instance({
            'sentences': sent_field,
            'categories': MultiLabelField(categories)
        })
