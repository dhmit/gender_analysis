from typing import Optional, Sequence, Dict, List, Union, TypedDict, Tuple
from statistics import median, mean

from gender_analysis.gender import BINARY_GROUP, Gender
from gender_analysis.text import Document
from gender_analysis.analysis.base_analyzers import CorpusAnalyzer
from gender_analysis.analysis.common import compute_bin_year


class DistanceStats(TypedDict):
    mean: float
    median: float
    min: int
    max: int


GenderDistanceStats = Dict[Gender, DistanceStats]
GenderDistances = Dict[Gender, List[int]]


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


def words_instance_dist(document: Document, target_words: List[str]) -> List[int]:
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


def _get_stats_from_distances(distance_results) -> DistanceStats:
    """
    Takes in a list of word distances and returns a DistanceStats dict
    :param distance_results: Instance Distance data as a list
    :return: dictionary of stats

    """
    stats: DistanceStats
    if len(distance_results) == 0:
        stats = {'median': 0, 'mean': 0, 'min': 0, 'max': 0}
    else:
        stats = {
            'median': median(distance_results),
            'mean': mean(distance_results),
            'min': min(distance_results),
            'max': max(distance_results)
        }

    return stats


def _get_stats_from_distances_by_metadata_value(
    distances_by_metadata_value: Dict[any, GenderDistances]
) -> Dict[any, GenderDistanceStats]:

    stats_by_metadata_value = {
        metadata_value: {} for metadata_value in distances_by_metadata_value
    }
    for metadata_value, distances_by_gender in distances_by_metadata_value.items():
        for gender, distances in distances_by_gender.items():
            stats_by_metadata_value[metadata_value][gender] = _get_stats_from_distances(distances)

    return stats_by_metadata_value




class GenderDistanceAnalyzer(CorpusAnalyzer):
    """
    The GenderDistanceAnalyzer finds the distance between occurances of sets of gendered
    pronouns. It can be used to compute the distance between,e.g., mentions of
    feminine vs. masculine pronouns in a corpus.

    Helper methods are provided to organize and analyze these distances using various averages
    and result filters based on document-specific metadata including the gender of a document's
    author, where the document was written, and in which year the document was written.

    In general, this analyzer class can return raw distance results,
    or stats for that set of results.

    For a small dataset, like our instance_distance test corpus, the raw results are usable.
    Here we see that some of our documents have no masculine pronouns in them,
    and we can eyeball that feminine pronouns occur with less distance between them.
    >>> from gender_analysis.analysis import GenderDistanceAnalyzer
    >>> from pathlib import Path
    >>> from gender_analysis.testing.common import TEST_DATA_DIR
    >>> data_dir = Path(TEST_DATA_DIR, 'instance_distance')
    >>> metadata_csv = Path(TEST_DATA_DIR, 'instance_distance', 'instance_distance_metadata.csv')
    >>> analyzer = GenderDistanceAnalyzer(file_path=data_dir, csv_path=metadata_csv)
    >>> analyzer.by_document()
    {<Document (instance_dist)>: {<Female>: [1, 2, 3, 4], <Male>: []}, <Document (instance_dist_masc)>: {<Female>: [], <Male>: [2, 3, 5, 7]}, <Document (words_instance_dist)>: {<Female>: [1, 2, 3, 4], <Male>: []}}

    We might want to aggregate these results across the whole corpus:
    >>> analyzer.corpus_results()
    {<Female>: [1, 1, 2, 2, 3, 3, 4, 4], <Male>: [2, 3, 5, 7]}

    Or take a look at corpus-wide statistics. Here we see what we suspected from the raw results:
    on average, feminine pronouns occur more densely in this corpus than masculine pronouns.
    >>> analyzer.corpus_stats()
    {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}

    We can also look at results or stats for every document:
    >>> analyzer.by_document()
    {<Document (instance_dist)>: {<Female>: [1, 2, 3, 4], <Male>: []}, <Document (instance_dist_masc)>: {<Female>: [], <Male>: [2, 3, 5, 7]}, <Document (words_instance_dist)>: {<Female>: [1, 2, 3, 4], <Male>: []}}

    >>> analyzer.stats_by_document()
    {<Document (instance_dist)>: {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}, <Document (instance_dist_masc)>: {<Female>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}, <Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}, <Document (words_instance_dist)>: {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}}

    You can filter on any Document metadata. Let's see if there are differences based on
    country of publication:
    analyzer.get_stats_by_metadata('country_publication')

    Or, for convenience, by the author's gender
    >>> analyzer.stats_by_author_gender()
    {'female': {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}, 'male': {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, <Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}}


    >>> analyzer.stats_by_date(time_frame=(1900, 1920), bin_size=10)
    """

    def __init__(self,
                 genders: Optional[Sequence[Gender]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if genders is None:
            genders = BINARY_GROUP
        self.genders = genders

        self._results = self.run_analysis()

    def run_analysis(self) -> Dict[Document, GenderDistances]:
        """
        Basic analysis of this module: computes GenderDistances per document per gender
        """
        results = {}

        for document in self.corpus:
            results[document] = {}
            for gender in self.genders:
                gender_results = words_instance_dist(document, gender.identifiers)
                results[document][gender] = gender_results

        return results

    def corpus_stats(self) -> GenderDistanceStats:
        corpus_distances_by_gender = self.corpus_results()

        corpus_stats = {}
        for gender, results_by_gender in corpus_distances_by_gender.items():
            corpus_stats[gender] = _get_stats_from_distances(results_by_gender)

        return corpus_stats

    def corpus_results(self) -> GenderDistances:
        corpus_distances_by_gender = {gender: [] for gender in self.genders}

        for document_results_by_gender in self._results.values():
            for gender, distances in document_results_by_gender.items():
                corpus_distances_by_gender[gender].extend(distances)

        for distances in corpus_distances_by_gender.values():
            distances.sort()

        return corpus_distances_by_gender

    def by_document(self):
        return self._results

    def stats_by_document(self) -> Dict[Document, GenderDistanceStats]:
        """
        Computes GenderDistanceStats per document

        :return: dict mapping Documents to GenderDistanceStats
        """
        return _get_stats_from_distances_by_metadata_value(self._results)

    def by_metadata(self, metadata_key: str) -> Dict[any, GenderDistances]:
        """
        Return GenderDistances organized by the values of a column from the Document metadata.
        Optionally, returns a single GenderDistances for a specific lookup value in that column.

        :param metadata_key: a string corresponding to one of the columns in the input metadata csv.

        Returns a dictionary mapping authors' genders to GenderDistances for documents
        they wrote, or a single GenderDistances for a specified author gender.
        """
        distances_by_metadata_value = {}
        for document, document_results_by_gender in self._results.items():
            # Skip the document if it doesn't the metadata we're looking for
            metadata_value = getattr(document, metadata_key, None)
            if metadata_value is None:
                continue

            # Init dict if metadata key is not already in the results dict
            if metadata_value not in distances_by_metadata_value:
                distances_by_metadata_value[metadata_value] = {
                    gender: [] for gender in self.genders
                }

            for gender, distances in document_results_by_gender.items():
                distances_by_metadata_value[metadata_value][gender].extend(distances)

        return distances_by_metadata_value

    def stats_by_metadata(
        self,
        metadata_key: str,
    ) -> Dict[any, GenderDistanceStats]:
        """
        Return GenderDistanceStats organized by the values of a column from the Document metadata.
        Optionally, returns a single GenderDistanceStats for a specific lookup value in that column.

        :param metadata_key: a string corresponding to one of the columns in the input metadata csv.

        Returns a dictionary mapping authors' genders to GenderDistanceStats for documents
        they wrote, or a single GenderDistanceStats for a specified author gender.
        """
        distances_by_metadata_value = self.by_metadata(metadata_key=metadata_key)
        return _get_stats_from_distances_by_metadata_value(distances_by_metadata_value)

    def by_author_gender(self) -> Dict[Gender, GenderDistances]:
        """
        Returns a dictionary mapping author_gender metadata to GenderDistances.
        This is a convenience method wrapper for by_metadata.
        """
        return self.by_metadata(metadata_key='author_gender')

    def stats_by_author_gender(self) -> Dict[Gender, GenderDistanceStats]:
        """
        Returns a dictionary mapping author_gender metadata to GenderDistanceStats.
        This is a convenience method wrapper for stats_by_metadata.
        """
        return self.stats_by_metadata(metadata_key='author_gender')

    def by_date(self,
                time_frame: Optional[Tuple[int, int]] = (-10000, 10000),
                bin_size: int = 1,
                ) -> Dict[int, GenderDistances]:
        """
        Organizes results by year.
        Optionally, can constrain those results to a given year range,
        and can bin results.
        """

        # n.b. we implement everything with bin and time_frame, but the defaults are set
        # so that you get results by year
        start_year, end_year = time_frame
        distances_by_year = self.by_metadata('date')

        distances_by_bin = {
            bin_start_year: {
                gender: [] for gender in self.genders
            }
            for bin_start_year in range(start_year, end_year, bin_size)
        }

        for year, distances_by_gender in distances_by_year.items():
            if start_year <= year < end_year:
                bin_start_year = compute_bin_year(year, start_year, end_year, bin_size)
                for gender in self.genders:
                    distances_by_bin[bin_start_year][gender].extend(distances_by_gender[gender])

        return distances_by_bin

    def stats_by_date(self,
                      time_frame: Optional[Tuple[int, int]] = (-10000, 10000),
                      bin_size: int = 1,
                      ) -> Dict[int, GenderDistanceStats]:

        distances_by_bin = self.by_date(time_frame=time_frame, bin_size=bin_size)
        return _get_stats_from_distances_by_metadata_value(distances_by_bin)

    def store(self, pickle_filepath: str = 'gendered_token_distance_analysis.pgz') -> None:
        """
        Saves self to a pickle file.

        :param pickle_filepath: filepath to save the output.
        :return: None, saves results as pickled file with name 'gender_tokens_analysis'
        """
        super().store(pickle_filepath=pickle_filepath)
