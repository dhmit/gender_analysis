from typing import Optional, Sequence, Dict, List, Union
from statistics import median, mean

from gender_analysis.gender import BINARY_GROUP, Gender
from gender_analysis.text import Document
from gender_analysis.analysis.base_analyzers import CorpusAnalyzer


def instance_dist(document: Document, word: str):
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


def words_instance_dist(document: Document, target_words: List[str]):
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
    """
    The GenderWordDistanceAnalyzer finds the distance between occurances of sets of gendered
    pronouns. It can be used to compute the distance between,e.g., mentions of
    feminine vs. masculine pronouns in a corpus.

    Helper methods are provided to organize and analyze these distances using various averages
    and result filters based on document-specific metadata including the gender of a document's
    author, where the document was written, and in which year the document was written.

    For a small dataset, like our instance_distance test corpus, the raw results are usable.
    Here we see that some of our documents have no masculine pronouns in them,
    and we can eyeball that feminine pronouns occur with less distance between them.
    >>> from gender_analysis.analysis import GenderWordDistanceAnalyzer
    >>> from pathlib import Path
    >>> from gender_analysis.testing.common import TEST_DATA_DIR
    >>> data_dir = Path(TEST_DATA_DIR, 'instance_distance')
    >>> metadata_csv = Path(TEST_DATA_DIR, 'instance_distance', 'instance_distance_metadata.csv')
    >>> analyzer = GenderWordDistanceAnalyzer(file_path=data_dir, csv_path=metadata_csv)
    >>> analyzer.get_results()
    {<Document (instance_dist)>: {<Female>: [1, 2, 3, 4], <Male>: []}, <Document (instance_dist_masc)>: {<Female>: [], <Male>: [2, 3, 5, 7]}, <Document (words_instance_dist)>: {<Female>: [1, 2, 3, 4], <Male>: []}}

    We might want to aggregate these results across the whole corpus:
    >>> analyzer.get_corpus_distances_by_gender()
    {<Female>: [1, 1, 2, 2, 3, 3, 4, 4], <Male>: [2, 3, 5, 7]}

    Or take a look at corpus-wide statistics. Here we see what we suspected from the raw results:
    on average, feminine pronouns occur more densely in this corpus than masculine pronouns.
    >>> analyzer.get_stats()
    {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}

    We can also look at stats for every document:
    >>> analyzer.get_stats_by_document()
    {<Document (instance_dist)>: {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}, <Document (instance_dist_masc)>: {<Female>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}, <Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}, <Document (words_instance_dist)>: {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}}

    Or a particular document:
    >>> analyzer.get_stats_by_document(analyzer.corpus.documents[0])
    {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}

    Several filters are available. For example, here we filter the results by the author's gender,
    to see whether there is a difference between the density of masculine and feminine pronouns
    depending on the author's gender.
    >>> analyzer.get_stats_by_author_gender()
    {'female': {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}, 'male': {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}}
    """

    def __init__(self,
                 genders: Optional[Sequence[Gender]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if genders is None:
            genders = BINARY_GROUP
        self.genders = genders

        self._results = self.run_analysis()

    def get_results(self):
        return self._results

    def run_analysis(self) -> Dict[Document, Dict[Gender, List[int]]]:
        results = {}

        for document in self.corpus:
            results[document] = {}
            for gender in self.genders:
                gender_results = words_instance_dist(document, gender.identifiers)
                results[document][gender] = gender_results

        return results

    def get_stats(self) -> Dict[Gender, Dict[str, float]]:
        corpus_distances_by_gender = self.get_corpus_distances_by_gender()

        corpus_stats = {}
        for gender, results_by_gender in corpus_distances_by_gender.items():
            corpus_stats[gender] = _get_stats(results_by_gender)

        return corpus_stats

    def get_corpus_distances_by_gender(self):
        corpus_distances_by_gender = {gender: [] for gender in self.genders}

        for document, document_results_by_gender in self._results.items():
            for gender, distances in document_results_by_gender.items():
                corpus_distances_by_gender[gender].extend(distances)

        for distances in corpus_distances_by_gender.values():
            distances.sort()

        return corpus_distances_by_gender

    def get_stats_by_document(self,
                              specific_document: Document = None
                              ) -> Union[Dict[Document, Dict[Gender, Dict[str, float]]],
                                         Dict[Gender, Dict[str, float]]]:
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
        :param specific_document: a specific document to get stats for
        :return: Nested dictionary, first mapping documents and next mapping genders to their stats.
        """
        results = {}

        for document in self.corpus:
            document_stats = dict()

            for gender in self.genders:
                gender_results = words_instance_dist(document, gender.identifiers)
                document_stats[gender] = _get_stats(gender_results)

            results[document] = document_stats

        if specific_document:
            return results[specific_document]
        else:
            return results

    def get_stats_by_author_gender(self, specific_gender: Gender = None):
        """
        Takes the raw analysis results and returns a dictionary in the following form:

        .. code-block:: python
            {
                'author_gender': {
                    <Gender object>: {
                        'median': float,
                        'mean: float,
                        'min': float,
                        'max': float
                    }
                }
            }
        """
        distances_by_author_gender = {}

        for document, document_results_by_gender in self._results.items():
            # Skip the document if it doesn't have a defined author gender
            author_gender = getattr(document, 'author_gender', None)
            if author_gender is None:
                continue

            # Init dict if author's gender is not already in the results dict
            if author_gender not in distances_by_author_gender:
                distances_by_author_gender[author_gender] = {
                    gender: [] for gender in self.genders
                }

            for gender, distances in document_results_by_gender.items():
                distances_by_author_gender[author_gender][gender].extend(distances)

        stats_by_author_gender = {
            author_gender: {} for author_gender in distances_by_author_gender
        }
        for author_gender, distances_by_gender in distances_by_author_gender.items():
            for gender, distances in distances_by_gender.items():
                stats_by_author_gender[author_gender][gender] = _get_stats(distances)

        return stats_by_author_gender


    def store(self, pickle_filepath: str = 'gendered_token_distance_analysis.pgz') -> None:
        """
        Saves self to a pickle file.

        :param pickle_filepath: filepath to save the output.
        :return: None, saves results as pickled file with name 'gender_tokens_analysis'
        """
        super().store(pickle_filepath=pickle_filepath)
