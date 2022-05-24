'''
This script is used to wrap method of Annotated Suffix Trees in convenient for
machine learning `fit` manner.
The initial algorithm is implemented in python by M.Dubov and improved by
A.Vlasov and D.Frolov.
https://github.com/dmitsf/AST-text-analysis
'''

import east
import pandas as pd
import numpy as np
from tqdm import tqdm

class AST:
    """
    Annotated Suffix Trees (AST)

    Given a collection of topics (key words) and a collection of texts
        computes a relevance matrix of each topic to each text using
        Annotated Trees built for texts.

    :param n_words: number of words to use for text-split into substrings
        (number of words in each substring)
    :type n_words: int
    """


    def __init__(self, n_words=5):
        tqdm.pandas()
        self.AST_trees = {}

        self.relevance_matrix = None
        self.topics_ast = None

        self.n_words = n_words

    def fit(self, texts, topics):
        """
        :param texts: list of texts to build trees for
        :type texts: list of str
        :param topics: list of topics for which you need to measure the
            relevance to texts using AS trees.
        :type topics: list of str
        """

        print("BUILDING AST'S FOR TEXTS")
        for i, text in enumerate(tqdm(texts)):
            if i not in self.AST_trees:
                ast = self.build_ast(text)
                self.AST_trees[i] = ast

        print("BUILDING relevance_matrix")
        self.relevance_matrix = np.empty((len(texts), len(topics)))
        self.topics_ast = self.preprocess_topics(topics)

        for i, ast in tqdm(self.AST_trees.items()):
            self.relevance_matrix[i] = np.array(self.score(ast))

    # @staticmethod
    def build_ast(self, text, n_words=5):
        return east.asts.base.AST.get_ast(
            east.utils.text_to_strings_collection(text, words=self.n_words)
        )

    def score(self, ast):
        return [ast.score(t) for t in self.topics_ast]

    @staticmethod
    def preprocess_topics(topics):
        topics_ast = []
        for topic in topics:
            topics_ast.append(
                east.utils.prepare_text(topic)
                    .replace(' ', '')
            )
        return topics_ast
