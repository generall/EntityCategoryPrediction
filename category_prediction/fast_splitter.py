from pprint import pprint
from typing import List

import re
from allennlp.data import Token
from allennlp.data.tokenizers.word_splitter import WordSplitter


@WordSplitter.register("fast-splitter")
class FastSplitter(WordSplitter):
    """
    Regular AllenNLP tokenizer if too slow.
    This one is simpler and faster
    """

    def __init__(self):
        self.pattern = re.compile(r'([\s()<>\[\]{\};:|\'\",./\\=\-])')

        self.space_re = re.compile(r'^\s*$')

    def split_words(self, sentence: str) -> List[Token]:
        """
        >>> FastSplitter().split_words("aaa,  asd @@sss@@@ sllls")
        [aaa, ,, asd, @@sss@@@, sllls]

        :param sentence:
        :return:
        """

        res = []
        offset = 0
        # print(sentence)

        pieces = re.split(self.pattern, sentence)

        for piece in pieces:
            if not re.match(self.space_re, piece):
                res.append(Token(piece, offset))
            offset += len(piece)

        # pprint(res)
        #
        # raise RuntimeError('stop')
        return res
