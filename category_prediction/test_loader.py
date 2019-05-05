from allennlp.data.tokenizers import WordTokenizer

from category_prediction import FastSplitter, FasttextTokenIndexer, MenionsLoader

if __name__ == '__main__':
    MenionsLoader(
        category_mapping_file='./data/test_category_mapping.json',
        token_indexers={
            "tokens": FasttextTokenIndexer(model_path="./data/fasttext_embedding.model",
                                           model_params_path="./data/fasttext_embedding.model.params")
        },
        tokenizer=WordTokenizer(
            word_splitter=FastSplitter()
        )
    )
