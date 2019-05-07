import time
from multiprocessing import Queue, Process

import tqdm
from allennlp.data.iterators.multiprocess_iterator import _queuer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer

from category_prediction import FastSplitter, FasttextTokenIndexer, MenionsLoader, StaticFasttextTokenIndexer

if __name__ == '__main__':

    indexer = StaticFasttextTokenIndexer(model_path="./data/fasttext_embedding.model",
                                   model_params_path="./data/fasttext_embedding.model.params")

    default_indexer = SingleIdTokenIndexer()

    loader = MenionsLoader(
        category_mapping_file='./data/test_category_mapping.json',
        token_indexers={
            "tokens": indexer
        },
        tokenizer=WordTokenizer(
            word_splitter=FastSplitter()
        )
    )

    input_queue = Queue(100)

    instances = loader.read('./data/train_data_aa.tsv')

    queuer = Process(target=_queuer, args=(instances, input_queue, 1, 10))
    queuer.start()

    time.sleep(1)


    def get_instances():
        instance = input_queue.get()
        while instance is not None:
            yield instance
            instance = input_queue.get()


    print("qsize", input_queue.qsize())

    for _ in tqdm.tqdm(get_instances()):
        pass

    queuer.join()
