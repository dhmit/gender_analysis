from gender_analysis.corpus import Corpus
from gender_analysis.testing import common


def test_load_large_corpus():
    """
    Tests that the `Corpus` class can load large corpora
    """

    c = Corpus(common.LARGE_TEST_CORPUS_PATH)

    assert len(c) == 99
    assert type(c.documents) == list
