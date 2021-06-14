from collections import Counter
from typing import Dict, Optional, Sequence, Tuple, Union
from gender_analysis.gender.common import BINARY_GROUP
from gender_analysis.gender import Gender
from gender_analysis.analysis.base_analyzers import CorpusAnalyzer

WordFrequency = Dict[str, float]
WordResponse = Union[Counter, WordFrequency]
GenderCounts = Dict[Gender, Counter]
GenderFrequencies = Dict[Gender, WordFrequency]


def _apply_display_preferences(gender_counts: GenderCounts,
                               format_by: str,
                               group_by: str,
                               total_word_count: int) -> Dict[Gender, Union[int, WordResponse]]:
    """
    A private helper function that applies the optional kwargs format_by and group_by to a
    Python dictionary keying Gender instances to Counter instances.

    :param gender_counts: a dictionary keying Gender instances to Counter instances.
    :param format_by: a string indicating the desired output of the identifier counts.
    :param group_by: a string indicating the method of aggregating counts.
    :param total_word_count: the total number of words in all texts used to constructr the
                             key_gender_dictionary.
    :return: a dictionary resembling the input but with display options applied.

    >>> from gender_analysis.gender.common import MALE, FEMALE
    >>> test_gender_dictionary = {MALE: {'he': 5, 'him': 2}, FEMALE: {'she': 3, 'her': 10}}
    >>> _apply_display_preferences(test_gender_dictionary, 'count', 'identifier', 100)
    {'Male': {'he': 5, 'him': 2}, 'Female': {'she': 3, 'her': 10}}
    >>> _apply_display_preferences(test_gender_dictionary, 'count', 'label', 100).get('Male')
    {'subject': 5, 'object': 2, 'other': 0}
    >>> _apply_display_preferences(test_gender_dictionary, 'count', 'aggregate', 100)
    {'Male': 7, 'Female': 13}
    >>> _apply_display_preferences(test_gender_dictionary, 'frequency', 'identifier', 100)
    {'Male': {'he': 0.05, 'him': 0.02}, 'Female': {'she': 0.03, 'her': 0.1}}
    >>> _apply_display_preferences(test_gender_dictionary, 'frequency', 'label', 100).get('Male')
    {'subject': 0.05, 'object': 0.02, 'other': 0}
    >>> _apply_display_preferences(test_gender_dictionary, 'frequency', 'aggregate', 100)
    {'Male': 0.07, 'Female': 0.13}
    >>> _apply_display_preferences(test_gender_dictionary, 'relative', 'identifier', 100)
    {'Male': {'he': 0.25, 'him': 0.1}, 'Female': {'she': 0.15, 'her': 0.5}}
    >>> _apply_display_preferences(test_gender_dictionary, 'relative', 'label', 100).get('Male')
    {'subject': 0.25, 'object': 0.1, 'other': 0}
    >>> _apply_display_preferences(test_gender_dictionary, 'relative', 'aggregate', 100)
    {'Male': 0.35, 'Female': 0.65}
    """

    if format_by == 'frequency':
        formatted_gender_counts = _get_gender_word_frequencies(gender_counts, total_word_count)
    elif format_by == 'relative':
        formatted_gender_counts = _get_gender_word_frequencies_relative(gender_counts)
    else:
        formatted_gender_counts = gender_counts

    output = {}

    for gender, counts in formatted_gender_counts.items():
        if group_by == 'label':
            output[gender.label] = _apply_identifier_labels(counts, gender)
        elif group_by == 'aggregate':
            output[gender.label] = sum(counts.values())
        else:
            output[gender.label] = counts

    return output


def _apply_identifier_labels(identifier_counts: WordResponse, gender: Gender) -> WordResponse:
    """
    A private helper function that accepts a dictionary of identifiers keyed to counts (either
    as integers or as floats) and aggregates those identifiers according to the given Gender's
    .subj and .obj properties.

    :param identifier_counts: a dictionary of identifiers keyed to integer or float counts.
    :param gender: a list containing instances of the Gender class.
    :return: a dictionary with all integer counts aggregated to the keys 'subject' and 'object.'

    >>> from gender_analysis.gender.common import FEMALE
    >>> test_input = {'she': 3, 'her': 10}
    >>> _apply_identifier_labels(test_input, FEMALE)
    {'subject': 3, 'object': 10, 'other': 0}
    """

    output = {'subject': 0, 'object': 0, 'other': 0}

    for identifier, count in identifier_counts.items():
        if identifier in gender.subj:
            output['subject'] += count
        elif identifier in gender.obj:
            output['object'] += count
        else:
            output['other'] += count

    return output


def _get_gender_word_frequencies(gender_word_counts: GenderCounts,
                                 total_word_count: int) -> GenderFrequencies:
    """
    A private helper function that copies and converts a dictionary of Gender instances keyed
    to identifier counts to a dictionary of Gender instances keyed to identifier frequencies.

    :param gender_word_counts: a dictionary keying Gender instances to Counter or dictionary
                               objects keying identifiers to counts.
    :return: a copy of the input with the integer count values replaced by floats indicating
             the count divided by the total_word_count.

    >>> from gender_analysis.gender.common import MALE, FEMALE
    >>> test_input = {MALE: Counter({'he': 2}), FEMALE: {'she': 4}}
    >>> _get_gender_word_frequencies(test_input, 20)
    {<Male>: {'he': 0.1}, <Female>: {'she': 0.2}}
    """

    output = {}
    for gender in gender_word_counts:
        output[gender] = {}
        for identifier in gender_word_counts[gender]:
            output[gender][identifier] = gender_word_counts[gender][identifier] / total_word_count
    return output


def _get_gender_word_frequencies_relative(gender_word_counts: GenderCounts) -> GenderFrequencies:
    """
    A private helper function that examines identifier counts keyed to Gender instances,
    determines the total count value of all identifiers across Gender instances,
    and returns the percentage of each identifier count over the total count.

    :param gender_word_counts: a dictionary keying gender instances to string identifiers keyed to
                               integer counts.
    :return: a dictionary with the integer counts transformed into float values representing
             the identifier count as a percentage of the total identifier counts across all
             identifiers.

    >>> from gender_analysis.gender.common import MALE, FEMALE
    >>> test_input = {MALE: {'he': 5, 'him': 2}, FEMALE: {'she': 3, 'her': 10}}
    >>> _get_gender_word_frequencies_relative(test_input)
    {<Male>: {'he': 0.25, 'him': 0.1}, <Female>: {'she': 0.15, 'her': 0.5}}
    """

    output = {}
    total_word_count = 0
    for gender in gender_word_counts:
        for word in gender_word_counts[gender]:
            total_word_count += gender_word_counts[gender][word]

    for gender in gender_word_counts:
        output[gender] = {}
        for word, original_count in gender_word_counts[gender].items():
            try:
                frequency = original_count / total_word_count
            except ZeroDivisionError:
                frequency = 0
            output[gender][word] = frequency

    return output


class GenderFrequencyAnalyzer(CorpusAnalyzer):
    """
    The GenderFrequencyAnalyzer instance accepts a series of texts and a series of Gender instances
    and finds occurrences of each of the Gender instances' identifiers (currently pronouns).
    Helper methods are provided to organize and analyze those occurrences according to
    relevant criteria.

    Instance methods:
        by_date()
        by_document()
        by_gender()
        by_identifier()
        by_metadata()
    """

    def __init__(self, genders: Optional[Sequence[Gender]] = None, **kwargs) -> None:
        """
        Initializes a GenderFrequencyAnalyzer object that can be used for retrieving
        analyses concerning the number of occurrences of gendered pronouns.

        GenderFrequencyAnalyzer is a subclass of CorpusAnalyzer and so accepts additional arguments
        from that class.

        CorpusAnalyzer params:
        :param corpus: an optional instance of the Corpus class.
        :param file_path: a filepath to .txt files for creating a Corpus instance.
        :param csv_path: a filepath to a .csv file containing metadata for .txt files.
        :param name: a string name to be passed to the Corpus instance.
        :param pickle_path: a filepath for writing the Corpus pickle file.
        :param ignore_warnings: a boolean value indicating whether or not warnings during Corpus
                                initialization should be displayed.

        GenderFrequencyAnalyzer params:
        :param genders: a list of Gender instances.
        """

        super().__init__(**kwargs)

        if genders is None:
            genders = BINARY_GROUP

        if not all(isinstance(item, Gender) for item in genders):
            raise ValueError('all items in list genders must be of type Gender')

        self.genders = genders

        count, frequencies, relatives = self._run_analysis()

        # memoized properties
        self._results_by_count = count
        self._results_by_frequencies = frequencies
        self._results_by_frequencies_relative = relatives
        self._by_gender = None

    def __str__(self):
        return "This is the Gender Frequency Analyzer for gendered pronouns."

    def _run_analysis(self):
        """
        A private helper method for running the primary analysis of GenderFrequencyAnalyzer.
        This method generates three dictionaries: one (count) keying Document instances
        to Gender instances to Counter instances representing the total number of instances
        of each Gender's identifiers in a given Document; one (frequency) keying Document instances
        to Gender instances to dictionaries of the shape {str:float} representing the total number
        of instances of each Gender's identifiers over the total word count of that Document; and
        one (relative) keying Document instances to Gender instances to dicationaries of the shape
        {str:float} representing the relative percentage of Gender identifiers across all Gender
        instances in a given Document instance.

        :return: a tuple containing three dictionaries.

        >>> from gender_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> from gender_analysis.text.corpus import Corpus
        >>> analyzer = GenderFrequencyAnalyzer(file_path=DOCUMENT_TEST_PATH,
        ...                                    csv_path=DOCUMENT_TEST_CSV)
        >>> analysis = analyzer._run_analysis()
        >>> test_document = analyzer.corpus.documents[0]
        >>> test_gender = analyzer.genders[0]
        >>> len(analysis) == 3
        True
        >>> list(analysis[0].keys()) == analyzer.corpus.documents
        True
        >>> list(analysis[1].keys()) == analyzer.corpus.documents
        True
        >>> list(analysis[2].keys()) == analyzer.corpus.documents
        True
        >>> list(analysis[0][test_document].keys()) == analyzer.genders
        True
        >>> list(analysis[1][test_document].keys()) == analyzer.genders
        True
        >>> list(analysis[2][test_document].keys()) == analyzer.genders
        True
        >>> isinstance(analysis[0][test_document][test_gender], Counter)
        True
        >>> isinstance(analysis[1][test_document][test_gender], dict)
        True
        >>> isinstance(analysis[2][test_document][test_gender], dict)
        True
        """

        count = {}
        frequencies = {}
        relatives = {}

        for document in self.corpus:
            count[document] = Counter()
            frequencies[document] = {}
            relatives[document] = {}
            for gender in self.genders:
                count[document][gender] = document.get_count_of_words(gender.identifiers)
                frequencies[document][gender] = document.get_word_frequencies(gender.identifiers)
            relatives[document] = _get_gender_word_frequencies_relative(count[document])

        return count, frequencies, relatives

    def by_date(self,
                time_frame: Tuple[int, int],
                bin_size: int,
                format_by: str = 'count',
                group_by: str = 'identifier'):
        """
        Return analysis organized by date (as determined by Document metadata).

        :param time_frame: a tuple of the format (start_date, end_date).
        :param bin_size: int for the number of years represented in each list of frequencies
        :param format_by: accepts 'frequency' and 'relative' as acceptable values, returns
                          analysis with counts as a frequency of all words in texts or
                          relative to one another, respectively.
        :param group_by: accepts 'label' and 'aggregate' as acceptable values, returns
                         analysis with counts grouped by pronoun category ('subject' and 'object')
                         or summed, respectively.
        :return: a dictionary of gender-pronoun pairs with top-level keys corresponding to the
                 values in the input Documents' 'date' metadata key.
        >>> from gender_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> from gender_analysis.text.corpus import Corpus
        >>> analyzer = GenderFrequencyAnalyzer(file_path=DOCUMENT_TEST_PATH,
        ...                                    csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer.by_date((2000, 2008), 2).keys()
        dict_keys([2000, 2002, 2004, 2006])
        >>> actual_analysis = analyzer.by_date((2000, 2010), 2).get(2002).get('Female')
        >>> expected_analysis = {'she': 0, 'her': 7, 'herself': 0, 'hers': 0}
        >>> actual_analysis == expected_analysis
        True
        """

        output = {}
        word_counts_by_bin_year = {}

        for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
            output[bin_start_year] = {gender: {} for gender in self.genders}
            word_counts_by_bin_year[bin_start_year] = 0

        for document in self.corpus:
            date = getattr(document, 'date', None)
            if date is None:
                continue
            bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
            if bin_year not in output:
                continue

            word_counts_by_bin_year[bin_year] += document.word_count

            for gender in self.genders:
                gender_identifier_counts = self._results_by_count[document][gender]
                if gender not in output[bin_year]:
                    output[bin_year][gender] = {}
                for identifier in gender.identifiers:
                    if identifier not in output[bin_year][gender]:
                        output[bin_year][gender][identifier] = 0
                    output[bin_year][gender][identifier] += gender_identifier_counts[identifier]

        for bin_year, gender_counts in output.items():
            output[bin_year] = _apply_display_preferences(gender_counts,
                                                          format_by,
                                                          group_by,
                                                          word_counts_by_bin_year[bin_year])

        return output

    def by_document(self, format_by: str = 'count', group_by: str = 'identifier'):
        """
        Return analysis organized by Document.label.

        :param format_by: accepts 'frequency' and 'relative' as acceptable values, returns
                          analysis with counts as a frequency of all words in texts or
                          relative to one another, respectively.
        :param group_by: accepts 'label' and 'aggregate' as acceptable values, returns
                         analysis with counts grouped by pronoun category ('subject' and 'object')
                         or summed, respectively.
        :return: a dictionary of gender-pronoun pairs with top-level keys corresponding to the
                 labels of the input Documents.

        >>> from gender_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderFrequencyAnalyzer(file_path=DOCUMENT_TEST_PATH,
        ...                                    csv_path=DOCUMENT_TEST_CSV)
        >>> doc = analyzer.corpus.documents[7]
        >>> analyzer_document_labels = list(analyzer.by_document().keys())
        >>> document_labels = list(map(lambda d: d.label, analyzer.corpus.documents))
        >>> analyzer_document_labels == document_labels
        True
        >>> expected_result = {'hers': 0, 'herself': 0, 'she': 0, 'her': 6}
        >>> actual_result = analyzer.by_document().get(doc.label).get('Female')
        >>> expected_result == actual_result
        True
        """

        output = {}

        for document in self.corpus:
            output[document.label] = _apply_display_preferences(self._results_by_count[document],
                                                                format_by,
                                                                group_by,
                                                                document.word_count)

        return output

    def by_gender(self, format_by: str = 'count', group_by: str = 'identifier'):
        """
        Return analysis organized by Gender.label.

        :param format_by: accepts 'frequency' and 'relative' as acceptable values, returns
                          analysis with counts as a frequency of all words in texts or
                          relative to one another, respectively.
        :param group_by: accepts 'label' and 'aggregate' as acceptable values, returns
                         analysis with counts grouped by pronoun category ('subject' and 'object')
                         or summed, respectively.
        :return: a dictionary of gender-pronoun pairs with top-level keys corresponding to the
                 labels of the input Genders.

        >>> from gender_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderFrequencyAnalyzer(file_path=DOCUMENT_TEST_PATH,
        ...                                    csv_path=DOCUMENT_TEST_CSV)
        >>> actual_female_results = Counter({'her': 14, 'she': 8, 'herself': 1, 'hers': 0})
        >>> actual_male_results = Counter({'his': 11, 'he': 11, 'him': 4, 'himself': 2})
        >>> expected_results = {'Male': actual_male_results, 'Female': actual_female_results}
        >>> actual_results = analyzer.by_gender()
        >>> actual_results == expected_results
        True
        """

        hashed_arguments = f"{str(format_by)}{str(group_by)}"

        if self._by_gender is None:
            self._by_gender = {hashed_arguments: None}
        elif hashed_arguments in self._by_gender:
            return self._by_gender[hashed_arguments]

        output = {
            gender: {
                identifier: 0 for identifier in gender.identifiers
            } for gender in self.genders
        }
        total_word_count = 0

        for document, gender_identifier_counts in self._results_by_count.items():
            total_word_count += document.word_count
            for gender, identifier_counts in gender_identifier_counts.items():
                for identifier, count in identifier_counts.items():
                    output[gender][identifier] += count

        output = _apply_display_preferences(output, format_by, group_by, total_word_count)

        self._by_gender[hashed_arguments] = output
        return output

    def by_identifier(self,
                      format_by: str = 'count',
                      group_by: str = 'identifier') -> Union[int, WordResponse]:
        """
        Return analysis organized by Gender identifiers.

        :param format_by: accepts 'frequency' and 'relative' as acceptable values, returns
                          analysis with counts as a frequency of all words in texts or
                          relative to one another, respectively.
        :param group_by: accepts 'label' and 'aggregate' as acceptable values, returns
                         analysis with counts grouped by pronoun category ('subject' and 'object')
                         or summed, respectively.
        :return: a dictionary of gender-pronoun pairs.

        >>> from gender_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderFrequencyAnalyzer(file_path=DOCUMENT_TEST_PATH,
        ...                                    csv_path=DOCUMENT_TEST_CSV)
        >>> actual_results = analyzer.by_identifier()
        >>> expected_results = {'she': 8, 'herself': 1, 'her': 14, 'hers': 0, 'him': 4, 'he': 11, \
                                'himself': 2, 'his': 11}
        >>> actual_results == expected_results
        True
        """

        results_by_gender = self.by_gender(format_by=format_by, group_by=group_by)

        if group_by == 'aggregate':
            output = 0
            for gender in results_by_gender:
                output += results_by_gender[gender]
        else:
            output = {}
            for gender in results_by_gender:
                for identifier in results_by_gender[gender]:
                    if identifier not in output:
                        output[identifier] = 0
                    output[identifier] += results_by_gender[gender][identifier]

        return output

    def by_metadata(self,
                    metadata_key: str,
                    format_by: str = 'count',
                    group_by: str = 'identifier') -> Dict[Union[str, int, float], WordResponse]:
        """
        Return analysis organized by input metadata_key (as determined by Document metadata).

        :param metadata_key: a string corresponding to one of the columns in the input metadata csv.
        :param format_by: accepts 'frequency' and 'relative' as acceptable values, returns
                          analysis with counts as a frequency of all words in texts or
                          relative to one another, respectively.
        :param group_by: accepts 'label' and 'aggregate' as acceptable values, returns
                         analysis with counts grouped by pronoun category ('subject' and 'object')
                         or summed, respectively.
        :return: a dictionary of gender-pronoun pairs with top-level keys corresponding to the
                 input metadata_key.

        >>> from gender_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderFrequencyAnalyzer(file_path=DOCUMENT_TEST_PATH,
        ...                                    csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer.by_metadata('author_gender').keys()
        dict_keys(['male', 'female'])
        >>> actual_results = analyzer.by_metadata('author_gender').get('male').get('Male')
        >>> expected_results = {'himself': 0, 'his': 4, 'him': 1, 'he': 7}
        >>> actual_results == expected_results
        True
        """

        output = {}
        word_counts_by_metadata_key = {}

        for document in self.corpus:
            matching_key = getattr(document, metadata_key, None)
            if matching_key is None:
                continue

            if matching_key not in output:
                output[matching_key] = {}

            if matching_key not in word_counts_by_metadata_key:
                word_counts_by_metadata_key[matching_key] = 0

            word_counts_by_metadata_key[matching_key] += document.word_count

            for gender in self.genders:
                if gender not in output[matching_key]:
                    output[matching_key][gender] = {}
                for identifier in gender.identifiers:
                    matching_identifier_count = self._results_by_count[document][gender][identifier]
                    if identifier not in output[matching_key][gender]:
                        output[matching_key][gender][identifier] = 0
                    output[matching_key][gender][identifier] += matching_identifier_count

        for key in output:
            output[key] = _apply_display_preferences(output[key],
                                                     format_by,
                                                     group_by,
                                                     word_counts_by_metadata_key[key])

        return output
