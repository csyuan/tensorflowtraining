#coding:utf-8
import numpy as np

class wordsprocessor(object):

    def __init__(self, max_length, file_path):
        self.max_length = max_length
        lines = list(open(file_path, "r").readlines())
        lines = [s.strip() for s in lines]
        self.vocabulary_ = dict([s.split("\t") for s in lines])

    def transform(self, raw_documents):
        """
        Transform documents to word-id matrix.
        """
        for tokens in raw_documents:
            word_ids = np.zeros(self.max_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_length:
                    break
                word_ids[idx] = self.get(token)
            yield word_ids

    def get(self, category):
        """Returns word's id in the vocabulary.
        Args:
          category: string or integer to lookup in vocabulary.

        Returns:
          interger, id in the vocabulary.
        """
        if category not in self.vocabulary_:
            return 0
        return self.vocabulary_[category]

# test
# word2id = wordsprocessor("./wordID.txt",3)
# hh = np.array(list(word2id.transform([["a","3","A"],["e","d","3"],["x","m","a"]])))


