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

    pattern = re.compile(r'([\s()<>\[\]{\};:|\'\",./\\=\-])')
    space_re = re.compile(r'^\s*$')

    def split_words(self, sentence: str) -> List[Token]:
        """
        >>> FastSplitter().split_words("aaa,  asd @@sss@@@ sllls")
        [aaa, ,, asd, @@sss@@@, sllls]

        :param sentence:
        :return:
        """

        res = []
        offset = 0
        # print(text)

        pieces = self.pattern.split(sentence)

        for piece in pieces:
            if not self.space_re.match(piece) and len(piece) > 0:
                res.append(Token(piece, offset))
            offset += len(piece)

        # pprint(res)
        #
        # raise RuntimeError('stop')
        return res
