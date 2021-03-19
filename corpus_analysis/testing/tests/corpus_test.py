from corpus_analysis.corpus import Corpus
from corpus_analysis.testing import common


class TestLoadCorpus:
    """
    Tests that the `Corpus` class can load corpora appropriately under different
    circumstances
    """

    def test_load_without_csv(self):
        """
        Tests that the corpus properly loads when not provided metadata
        """

        c = Corpus(common.TEST_CORPUS_PATH)
        assert len(c) == 99
        assert type(c.documents) == list
        assert c.name is None

    def test_load_with_csv(self):
        """
        Test that the corpus properly loads when provided a metadata csv
        """

        c = Corpus(
            common.TEST_CORPUS_PATH,
            csv_path=common.LARGE_TEST_CORPUS_CSV,
            name='test_corpus',
        )
        assert len(c) == 99
        assert type(c.documents) == list
        assert c.name == 'test_corpus'

    def test_load_pickle(self, tmp_path):
        """
        Tests that the corpus can properly load from a pickle file, while retaining
        all of the relevant information

        :param tmp_path: a temporary directory created by pytest that will be used to store
            a pickle file from the test
        """

        pickle_path = tmp_path / 'pickle.pgz'

        original_corpus = Corpus(
            common.TEST_CORPUS_PATH,
            csv_path=common.SMALL_TEST_CORPUS_CSV,
            name='test_corpus',
            pickle_on_load=pickle_path,
            ignore_warnings=True
        )

        # first make sure the small corpus is correct
        assert len(original_corpus) == 10
        assert type(original_corpus.documents) == list
        assert original_corpus.name == 'test_corpus'

        # next load the pickle file to make sure data was copied correctly
        pickle_corpus = Corpus(pickle_path, name='test_corpus')
        assert len(pickle_corpus) == 10
        assert type(original_corpus.documents) == list
        assert pickle_corpus.name == 'test_corpus'

        # Make sure the corpora are equal
        assert original_corpus == pickle_corpus
