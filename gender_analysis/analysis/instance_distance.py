from typing import Optional, Sequence, Dict
from statistics import median, mean

from gender_analysis.gender import common, Gender
from gender_analysis.text import Document
from gender_analysis.analysis.base_analyzers import CorpusAnalyzer


def instance_dist(document, word):
    """
    Takes in a word and returns a list of distances between each instance of that word in
    the document, where distance means 'number of words' between occurences of the target word.

    :param document: Document object to analyze
    :param word: str, word to search text for distance between occurrences
    :return: list of distances between consecutive instances of word

    >>> from gender_analysis import Document
    >>> from pathlib import Path
    >>> from gender_analysis.testing.common import TEST_DATA_DIR
    >>> document_metadata = {
    ...    'filepath': Path(TEST_DATA_DIR, 'instance_distance', 'instance_dist.txt'),
    ...    'filename': 'instance_dist.txt'
    ... }
    >>> doc = Document(document_metadata)
    >>> instance_dist(doc, 'her')
    [1, 2, 3, 4]

    """
    return words_instance_dist(document, [word])


def words_instance_dist(document, target_words):
    """
    Takes in a document and list of words (e.g. gendered pronouns), returns a list of distances
    between each instance of any of the words in that document,
    where a distance of N means that the second word is the Nth word after the initial word

    :param document: Document object
    :param target_words: list of strings; words to search for in the document
    :return: list of distances between instances of any word in the supplied list

    >>> from gender_analysis import Document
    >>> from pathlib import Path
    >>> from gender_analysis.testing.common import TEST_DATA_DIR
    >>> document_metadata = {
    ...    'filepath': Path(TEST_DATA_DIR, 'instance_distance', 'words_instance_dist.txt'),
    ...    'filename': 'words_instance_dist.txt'
    ... }
    >>> doc = Document(document_metadata)
    >>> words_instance_dist(doc, ['she', 'her', 'hers', 'herself'])
    [1, 2, 3, 4]
    """
    text = document.get_tokenized_text()
    distances = []
    distance = 0
    found_first_word = False

    for token in text:
        if not found_first_word:
            if token in target_words:
                found_first_word = True
        else:
            distance += 1
            if token in target_words:
                distances.append(distance)
                distance = 0
    return distances


def _get_stats(distance_results):
    """
    Returns a dictionary of data from a given set of instance distance data, with keys of 'median',
    'mean', 'min', and 'max'.

    :param distance_results: Instance Distance data as a list
    :return: dictionary of stats

    """
    if len(distance_results) == 0:
        return {'median': 0, 'mean': 0, 'min': 0, 'max': 0}
    else:
        return {
            'median': median(distance_results),
            'mean': mean(distance_results),
            'min': min(distance_results),
            'max': max(distance_results)
        }


def _get_document_gender_metrics(document_results, metric):
    """
    This is a helper function for ``results_by`` functions.

    Pulls out the given metric from a document's min/max/etc. dictionary and maps each gender
    as a key.

    :param document_results: A dictionary mapping different fields to a dictionary of min/max/etc.
    :param metric: The metric to extract from results
    :return: A dictionary mapping each gender from ``document_results`` to the value of the metric
        for that field
    """

    metric_list = dict()
    for field in document_results:
        metric_list[field] = document_results[field][metric]

    return metric_list


def results_by_author_gender(results, metric):
    """
    Takes in a dictionary of results and a specified metric from ``run_distance_analysis``, returns
    a dictionary in the following form:

    .. code-block:: python

        {
            'author_gender': {
                <Document Object>: {
                    <Gender Object>: metric_result
                }
            }
        }


    :param results: dictionary from **run_distance_analysis**
    :param metric: one of ('median', 'mean', 'min', 'max')

    :return: dictionary

    """
    data = dict()
    if metric not in {'median', 'mean', 'min', 'max'}:
        raise ValueError(
            f"{metric} is not valid metric name. Valid names: 'median', 'mean', 'min', 'max'"
        )

    for document in results:

        # Skip the document if it doesn't have a defined author gender
        author_gender = getattr(document, 'author_gender', None)
        if author_gender is None:
            continue

        if author_gender not in data:
            data[author_gender] = dict()

        data[author_gender][document] = _get_document_gender_metrics(results[document], metric)

    return data


def results_by_year(results, metric, time_frame, bin_size):
    """
    Takes in a dictionary of results and a specified metric from ``run_distance_analysis``, returns
    a dictionary in the following form:

    .. code-block:: python

        {
            year: {
                <Document Object>: {
                    <Gender Object>: metric_result
                }
            }
        }

    :param results: dictionary
    :param metric: ('median', 'mean', 'min', 'max')
    :param time_frame: tuple (int start year, int end year) for the range of dates to return
        frequencies
    :param bin_size: int for the number of years represented in each document dictionary

    :return: nested dictionary, mapping integer years->`Document`s->`Gender`s->float results

    """

    if metric not in {'median', 'mean', 'min', 'max'}:
        raise ValueError(
            f"{metric} is not valid metric name. Valid names: 'median', 'mean', 'min', 'max'"
        )

    data = {}
    for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
        data[bin_start_year] = dict()

    for document in results.keys():

        date = getattr(document, 'date', None)
        if date is None:
            continue

        bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
        data[bin_year][document] = _get_document_gender_metrics(results[document], metric)

    return data


def results_by_location(results, metric):
    """
    Takes in a dictionary of results and a specified metric from ``run_distance_analysis``, returns
    a dictionary of the following form:

    .. code-block:: python

        {
            'location': {
                <Document Object>: {
                    <Gender Object>: metric_result
                }
            }
        }

    :param results: dictionary
    :param metric: ('median', 'mean', 'min', 'max')

    :return: dictionary

    """
    data = {}
    metric_indexes = {"median": 0, "mean": 2, "min": 3, "max": 4}
    if metric not in metric_indexes:
        raise ValueError(
            f"{metric} is not valid metric name. Valid names: 'median', 'mean', 'min', 'max'"
        )

    for document in results.keys():

        location = getattr(document, 'country_publication', None)
        if location is None:
            continue

        if location not in data:
            data[location] = dict()

        data[location][document] = _get_document_gender_metrics(results[document], metric)

    return data


def get_highest_distances(results, num):
    """
    Finds the documents with the largest median distances between each gender that was analyzed.

    Returns a dictionary mapping genders to a list of tuples of the form (``Document``, median),
    where ``Document``\\ s with higher medians are listed first.

    :param results: dictionary of results from ``run_distance_analysis``
    :param num: number of top distances to return

    :return: Dictionary of lists of tuples.
    """

    medians = dict()

    # Get all of the medians for the documents
    for document in results:
        for gender in results['document']:
            if gender not in medians:
                medians[gender] = list()

            medians[gender].append((results[document][gender]['median'], document))

    # Pull out the top medians
    top_medians = dict()
    for gender in medians:
        top_medians[gender] = sorted(medians[gender], reverse=True)[0:num]

    return top_medians


class GenderWordDistanceAnalyzer(CorpusAnalyzer):
    def __init__(self, genders: Optional[Sequence[Gender]] = None):
        super().__init__(**kwargs)

        if genders is None:
            genders = BINARY_GROUP
        self.genders = genders

        self._results = self.run_analysis()

    def get_results(self) -> Dict[Document, Dict[Gender, Dict[str, float]]]:
        return self._results

    def run_analysis() -> Dict[Document, Dict[Gender, Dict[str, float]]]:
        """
        Returns a dictionary with each document in the Analyzer's corpus
        mapped in the following form:

        .. code-block:: python

            {
                <Document object>: {
                    <Gender object>: {
                        'median': float,
                        'mean: float,
                        'min': float,
                        'max': float
                    }
                }
            }

        :return: Nested dictionary, first mapping documents and next mapping genders to their stats.
        """
        results = {}

        for document in self.corpus:
            document_stats = dict()

            for gender in self.genders:
                gender_results = words_instance_dist(document, gender.identifiers)
                document_stats[gender] = _get_stats(gender_results)

            results[document] = document_stats

        return results

    def store(self, pickle_filepath: str = 'gendered_token_distance_analysis.pgz') -> None:
        """
        Saves self to a pickle file.

        :param pickle_filepath: filepath to save the output.
        :return: None, saves results as pickled file with name 'gender_tokens_analysis'
        """
        super().store(pickle_filepath=pickle_filepath)
