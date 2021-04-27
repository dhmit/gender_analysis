from typing import Optional
from collections import Counter
from .corpus import Corpus

from gender_analysis.common import SWORDS_ENG

class Analyzer:
    """
    Base analyzer class.
    Handles initialization of Corpus and most common analysis methods.
    Intended for topic-specific subclassing.
    """

    def __init__(
        self,
        corpus: Optional[Corpus] = None,
        file_path: str = None,
        csv_path: str = None,
        name: str = None,
        pickle_path: str = None,
        ignore_warnings: bool = False,
    ):
        if corpus:
            self.corpus = corpus
        else:
            if not (file_path and csv_path):
                # TODO(ra): real exception type, better message
                raise Exception('You must pass a file_path and csv_path')

            # Create a new corpus
            self.corpus = Corpus(
                file_path,
                csv_path=csv_path,
                name=name,
                pickle_path=pickle_path,
                ignore_warnings=ignore_warnings,
            )

    def get_word_count(self, remove_swords: bool = False) -> Counter:
        """
        This function returns a Counter object that stores
        how many times each word appears in the corpus.

        :return: Python Counter object

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import (
        ...     TEST_CORPUS_PATH as path,
        ...     SMALL_TEST_CORPUS_CSV as path_to_csv
        ... )
        >>> c = Corpus(path, csv_path=path_to_csv, ignore_warnings = True)
        >>> word_count = c.get_wordcount_counter()
        >>> word_count['fire']
        157

        """
        corpus_counter = Counter()
        for document in self.corpus:
            document_counter = document.get_wordcount_counter()
            corpus_counter += document_counter
        if remove_swords:
            for word in list(corpus_counter):
                if word in SWORDS_ENG:
                    del corpus_counter[word]
        return corpus_counter
