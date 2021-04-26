from typing import Optional
from .corpus import Corpus


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
