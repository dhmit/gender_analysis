from pathlib import Path
import pytest
from corpus_analysis import Corpus, Document
from gender_analysis import common
from gender_analysis.analysis.instance_distance import run_distance_analysis, \
    get_highest_distances, results_by_location, results_by_author_gender, results_by_year
from gender_analysis.testing import common as test_common

documents = []
corpus = Corpus(test_common.DOCUMENT_TEST_PATH, csv_path=test_common.DOCUMENT_TEST_CSV)
for i in range(14):
    name = "test_text_" + str(i) + ".txt"
    location = "USA" if i % 2 == 0 else "UK"
    author_gender = "Male" if i % 3 == 0 else "Female"
    date = i % 10 + 1950
    documents.append(Document({"filename": name, "country_publication": location, "date": str(date),
                               "author_gender": author_gender,
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

    def test_results_by_location(self):
        results = run_distance_analysis(corpus)

        results = results_by_location(results, "median")

        expected = dict()
        expected["USA"] = {}
        expected["UK"] = {}
        expected["USA"][documents[4]] = {common.MALE: 6, common.FEMALE: 0}
        expected["UK"][documents[11]] = {common.MALE: 2, common.FEMALE: 17.5}
        expected["UK"][documents[3]] = {common.MALE: 0, common.FEMALE: 6}


        assert results["UK"][documents[11]] == expected["UK"][documents[11]]
        assert results["UK"][documents[3]] == expected["UK"][documents[3]]
        assert results["USA"][documents[4]] == expected["USA"][documents[4]]

    def test_results_by_author_gender(self):
        results = run_distance_analysis(corpus)
        results = results_by_author_gender(results, "median")

        expected = dict()
        expected["male"] = {}
        expected["female"] = {}
        expected["female"][documents[4]] = {common.MALE: 6, common.FEMALE: 0}
        expected["female"][documents[11]] = {common.MALE: 2, common.FEMALE: 17.5}
        expected["male"][documents[3]] = {common.MALE: 0, common.FEMALE: 6}


        assert results["female"][documents[11]] == expected["female"][documents[11]]
        assert results["male"][documents[3]] == expected["male"][documents[3]]
        assert results["female"][documents[4]] == expected["female"][documents[4]]

    def test_results_by_date(self):
        results = run_distance_analysis(corpus)

        results = results_by_year(results, "median", (1950, 1962), 1)
        results[1953].pop(documents[13])

        expected = dict()
        expected[1953] = {}
        expected[1954] = {}
        expected[1954][documents[4]] = {common.MALE: 6, common.FEMALE: 0}
        expected[1953][documents[3]] = {common.MALE: 0, common.FEMALE: 6}

        assert results[1953] == expected[1953]
        assert results[1954] == expected[1954]

    def test_results_error_throwing(self):
        results = run_distance_analysis(corpus)
        with pytest.raises(ValueError):
            results2 = results_by_year(results, "mode", (1950, 1962), 1)

        with pytest.raises(ValueError):
            results2 = results_by_location(results, "mode")

        with pytest.raises(ValueError):
            results2 = results_by_author_gender(results, "mode")
