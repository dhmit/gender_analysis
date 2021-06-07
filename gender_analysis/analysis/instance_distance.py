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
    {'instance_dist': {<Female>: [1, 2, 3, 4], <Male>: []}, \
'instance_dist_masc': {<Female>: [], <Male>: [2, 3, 5, 7]}, \
'words_instance_dist': {<Female>: [1, 2, 3, 4], <Male>: []}}

    We might want to aggregate these results across the whole corpus:
    >>> analyzer.corpus_results()
    {<Female>: [1, 1, 2, 2, 3, 3, 4, 4], <Male>: [2, 3, 5, 7]}

    Or take a look at corpus-wide statistics. Here we see what we suspected from the raw results:
    on average, feminine pronouns occur more densely in this corpus than masculine pronouns.
    >>> analyzer.corpus_stats()
    {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}

    We can also look at results or stats for every document:
    >>> analyzer.by_document()
    {'instance_dist': {<Female>: [1, 2, 3, 4], <Male>: []}, \
'instance_dist_masc': {<Female>: [], <Male>: [2, 3, 5, 7]}, \
'words_instance_dist': {<Female>: [1, 2, 3, 4], <Male>: []}}

    >>> analyzer.stats_by_document()
    {'instance_dist': \
{<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}, \
'instance_dist_masc': \
{<Female>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}, \
<Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}, \
'words_instance_dist': \
{<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}}
    """
    def __init__(self,
                 genders: Optional[Sequence[Gender]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        if genders is None:
            genders = BINARY_GROUP
        self.genders = genders

        self._results = self._run_analysis()

    def _run_analysis(self) -> Dict[Document, GenderDistances]:
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

    def corpus_results(self) -> GenderDistances:
        """
        Aggregates distance results across the whole corpus

        See doctest for class for usage.
        """
        corpus_distances_by_gender = {gender: [] for gender in self.genders}

        for document_results_by_gender in self._results.values():
            for gender, distances in document_results_by_gender.items():
                corpus_distances_by_gender[gender].extend(distances)

        for distances in corpus_distances_by_gender.values():
            distances.sort()

        return corpus_distances_by_gender

    def corpus_stats(self) -> GenderDistanceStats:
        """
        Aggregates stats across the whole corpus

        See doctest for class for usage.
        """
        corpus_distances_by_gender = self.corpus_results()

        corpus_stats = {}
        for gender, results_by_gender in corpus_distances_by_gender.items():
            corpus_stats[gender] = _get_stats_from_distances(results_by_gender)

        return corpus_stats

    def by_document(self) -> Dict[str, GenderDistances]:
        """
        Organizes results by document label

        See doctest for class for usage.
        """
        return {
            document.label: distances for document, distances in self._results.items()
        }

    def stats_by_document(self) -> Dict[str, GenderDistanceStats]:
        """
        Organizes stats by document label

        See doctest for class for usage.
        """
        return _get_stats_from_distances_by_metadata_value(self.by_document())

    def by_metadata(self, metadata_key: str) -> Dict[any, GenderDistances]:
        """
        Aggregates GenderDistances based on the values of a specified Document metadata field.

        :param metadata_key: a string corresponding to one of the columns in the input metadata csv.
        >>> from gender_analysis.analysis import GenderDistanceAnalyzer
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> data_dir = Path(TEST_DATA_DIR, 'instance_distance')
        >>> metadata_csv = Path(TEST_DATA_DIR, 'instance_distance', 'instance_distance_metadata.csv')
        >>> analyzer = GenderDistanceAnalyzer(file_path=data_dir, csv_path=metadata_csv)
        >>> analyzer.by_metadata('country_publication')
        {'United States': {<Female>: [1, 2, 3, 4], <Male>: [2, 3, 5, 7]}, \
'Canada': {<Female>: [1, 2, 3, 4], <Male>: []}}
        >>> analyzer.by_metadata('country_publication').get('United States')
        {<Female>: [1, 2, 3, 4], <Male>: [2, 3, 5, 7]}
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
        Aggregates GenderDistancesStats based on the values of a specified Document metadata field.

        :param metadata_key: a string corresponding to one of the columns in the input metadata csv.

        >>> from gender_analysis.analysis import GenderDistanceAnalyzer
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> data_dir = Path(TEST_DATA_DIR, 'instance_distance')
        >>> metadata_csv = Path(TEST_DATA_DIR, 'instance_distance', 'instance_distance_metadata.csv')
        >>> analyzer = GenderDistanceAnalyzer(file_path=data_dir, csv_path=metadata_csv)
        >>> analyzer.stats_by_metadata('country_publication')
        {'United States': \
{<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}, \
'Canada': \
{<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}}

        >>> analyzer.stats_by_metadata('country_publication').get('United States')
        {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}
        """
        distances_by_metadata_value = self.by_metadata(metadata_key=metadata_key)
        return _get_stats_from_distances_by_metadata_value(distances_by_metadata_value)

    def by_author_gender(self) -> Dict[str, GenderDistances]:
        """
        Organizes results by the 'author_gender' metadata on each Document.
        This is a convenience method wrapper for by_metadata.

        >>> from gender_analysis.analysis import GenderDistanceAnalyzer
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> data_dir = Path(TEST_DATA_DIR, 'instance_distance')
        >>> metadata_csv = Path(TEST_DATA_DIR, 'instance_distance', 'instance_distance_metadata.csv')
        >>> analyzer = GenderDistanceAnalyzer(file_path=data_dir, csv_path=metadata_csv)
        >>> analyzer.by_author_gender()
        {'female': {<Female>: [1, 2, 3, 4], <Male>: [2, 3, 5, 7]}, \
'male': {<Female>: [1, 2, 3, 4], <Male>: []}}

        >>> analyzer.by_author_gender().get('female')
        {<Female>: [1, 2, 3, 4], <Male>: [2, 3, 5, 7]}
        """
        return self.by_metadata(metadata_key='author_gender')

    def stats_by_author_gender(self) -> Dict[str, GenderDistanceStats]:
        """
        Organizes results by the 'author_gender' metadata on each Document.
        This is a convenience method wrapper for stats_by_metadata.

        >>> from gender_analysis.analysis import GenderDistanceAnalyzer
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> data_dir = Path(TEST_DATA_DIR, 'instance_distance')
        >>> metadata_csv = Path(TEST_DATA_DIR, 'instance_distance', 'instance_distance_metadata.csv')
        >>> analyzer = GenderDistanceAnalyzer(file_path=data_dir, csv_path=metadata_csv)
        >>> analyzer.stats_by_author_gender()
        {'female': \
{<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}, \
'male': \
{<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}}

        >>> analyzer.stats_by_author_gender().get('female')
        {<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}
        """
        return self.stats_by_metadata(metadata_key='author_gender')

    def by_date(self,
                time_frame: Optional[Tuple[int, int]] = None,
                bin_size: int = 1,
                ) -> Dict[int, GenderDistances]:
        """
        Organizes results by the 'date' metadata on each Document.
        Optionally, can constrain those results to a given year range and can bin results.

        >>> from gender_analysis.analysis import GenderDistanceAnalyzer
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> data_dir = Path(TEST_DATA_DIR, 'instance_distance')
        >>> metadata_csv = Path(TEST_DATA_DIR, 'instance_distance', 'instance_distance_metadata.csv')
        >>> analyzer = GenderDistanceAnalyzer(file_path=data_dir, csv_path=metadata_csv)
        >>> analyzer.by_date()
        {1900: {<Female>: [1, 2, 3, 4], <Male>: []}, \
1901: {<Female>: [], <Male>: [2, 3, 5, 7]}, \
1910: {<Female>: [1, 2, 3, 4], <Male>: []}}
        >>> analyzer.by_date(time_frame=(1900, 1920), bin_size=10)
        {1900: {<Female>: [1, 2, 3, 4], <Male>: [2, 3, 5, 7]}, \
1910: {<Female>: [1, 2, 3, 4], <Male>: []}}
        """

        if bin_size != 1 and time_frame is None:
            raise ValueError('Bin sizes greater than 1 require a time_frame to be set')
        if bin_size < 1:
            raise ValueError('Bin size must be at least 1.')

        distances_by_year = self.by_metadata('date')
        distances_by_bin = {}
        for year, distances_by_gender in distances_by_year.items():
            if time_frame and not (time_frame[0] <= year < time_frame[1]):
                # this year is not within the timeframe
                continue

            if bin_size == 1:
                bin_start_year = year
            else:
                bin_start_year = compute_bin_year(year, time_frame[0], time_frame[1], bin_size)

            if bin_start_year not in distances_by_bin:
                distances_by_bin[bin_start_year] = {
                    gender: [] for gender in self.genders
                }

            for gender in self.genders:
                distances_by_bin[bin_start_year][gender].extend(distances_by_gender[gender])

        return distances_by_bin

    def stats_by_date(self,
                      time_frame: Optional[Tuple[int, int]] = None,
                      bin_size: int = 1,
                      ) -> Dict[int, GenderDistanceStats]:
        """
        Organizes stats by the 'date' metadata on each Document.
        Optionally, can constrain those results to a given year range and can bin results.

        >>> from gender_analysis.analysis import GenderDistanceAnalyzer
        >>> from pathlib import Path
        >>> from gender_analysis.testing.common import TEST_DATA_DIR
        >>> data_dir = Path(TEST_DATA_DIR, 'instance_distance')
        >>> metadata_csv = Path(TEST_DATA_DIR, 'instance_distance', 'instance_distance_metadata.csv')
        >>> analyzer = GenderDistanceAnalyzer(file_path=data_dir, csv_path=metadata_csv)
        >>> analyzer.stats_by_date()
        {1900: \
{<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}, \
1901: \
{<Female>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}, \
<Male>: {'median': 4.0, 'mean': 4.25, 'min': 2, 'max': 7}}, \
1910: \
{<Female>: {'median': 2.5, 'mean': 2.5, 'min': 1, 'max': 4}, \
<Male>: {'median': 0, 'mean': 0, 'min': 0, 'max': 0}}}
        """
        distances_by_bin = self.by_date(time_frame=time_frame, bin_size=bin_size)
        return _get_stats_from_distances_by_metadata_value(distances_by_bin)

    def store(self, pickle_filepath: str = 'gendered_token_distance_analysis.pgz') -> None:
        """
        Saves self to a pickle file.

        :param pickle_filepath: filepath to save the output.
        :return: None, saves results as pickled file with name 'gender_tokens_analysis'
        """
        super().store(pickle_filepath=pickle_filepath)
