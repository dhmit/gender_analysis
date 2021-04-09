from pathlib import Path

from corpus_analysis import Corpus, Document
from gender_analysis import common
from gender_analysis.analysis import run_distance_analysis, get_highest_distances
from gender_analysis.testing import common as test_common

documents = []
corpus = Corpus(test_common.DOCUMENT_TEST_PATH, csv_path=test_common.DOCUMENT_TEST_CSV)
for i in range(14):
    name = "test_text_" + str(i) + ".txt"
    documents.append(Document({"filename": name,
                     "filepath": Path(test_common.DOCUMENT_TEST_PATH, name)}))


class TestInstanceDistance:

    def test_simple_run_distance_analysis_male(self):
        results = run_distance_analysis(corpus, {common.MALE})

        expected = dict()

        expected[documents[1]] = {}
        expected[documents[1]][common.MALE] = {"mean": 0, "median": 0, "min": 0, "max": 0}

        expected[documents[4]] = {}
        expected[documents[4]][common.MALE] = {"mean": 6, "median": 6, "min": 5, "max": 7}

        expected[documents[3]] = {}
        expected[documents[3]][common.MALE] = {"mean": 0, "median": 0, "min": 0, "max": 0}

        assert expected[documents[1]] == results[documents[1]]
        assert expected[documents[3]] == results[documents[3]]
        assert expected[documents[4]] == results[documents[4]]

    def test_simple_run_distance_analysis_female(self):
        results = run_distance_analysis(corpus, {common.FEMALE})

        expected = dict()

        expected[documents[1]] = {}
        expected[documents[1]][common.FEMALE] = {"mean": 0, "median": 0, "min": 0, "max": 0}

        expected[documents[4]] = {}
        expected[documents[4]][common.FEMALE] = {"mean": 0, "median": 0, "min": 0, "max": 0}

        expected[documents[3]] = {}
        expected[documents[3]][common.FEMALE] = {"mean": 6.2, "median": 6, "min": 5, "max": 7}

        assert expected[documents[1]] == results[documents[1]]
        assert expected[documents[3]] == results[documents[3]]
        assert expected[documents[4]] == results[documents[4]]

    def test_simple_run_distance_analysis_unspecified(self):
        results = run_distance_analysis(corpus)

        expected = dict()

        expected[documents[11]] = {}
        expected[documents[11]][common.FEMALE] = {"mean": 17.5, "median": 17.5, "min": 6, "max": 29}
        expected[documents[11]][common.MALE] = {"mean": 5, "median": 2, "min": 1, "max": 12}

        assert expected[documents[11]] == results[documents[11]]

    def test_get_highest_distances(self):
        results = run_distance_analysis(corpus)
        results = get_highest_distances(results, 14)
        print(results)
        i = 0
        while i < len(results[common.MALE]):
            if results[common.MALE][i][1] not in (documents[4], documents[3], documents[11]):
                results[common.MALE].remove(results[common.MALE][i])
            else:
                i += 1

        i = 0
        while i < len(results[common.FEMALE]):
            if results[common.FEMALE][i][1] not in (documents[4], documents[3], documents[11]):
                results[common.FEMALE].remove(results[common.FEMALE][i])
            else:
                i += 1

        assert results == {common.MALE: [(6, documents[4]), (2, documents[11]), (0, documents[3])],
                           common.FEMALE: [(17.5, documents[11]), (6, documents[3]),
                                           (0, documents[4])]}
