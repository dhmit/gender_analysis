from collections import Counter
from typing import Dict, Optional, Sequence, Tuple, Union
from functools import reduce
from more_itertools import windowed
import nltk

from corpus_analysis.analyzer import Analyzer
from corpus_analysis.common import load_pickle, store_pickle
from corpus_analysis.document import Document

from gender_analysis.common import MALE, FEMALE, BINARY_GROUP, SWORDS_ENG
from gender_analysis.gender import Gender


NLTK_ADJECTIVE_TOKENS = ["JJ", "JJR", "JJS"]
NLTK_NOUN_TOKENS = ['NN']


GenderTokenCounters = Dict[str, Counter]


GenderTokenSequence = Dict[str, Sequence[Tuple[str, int]]]


GenderTokenResponse = Union[GenderTokenCounters, GenderTokenSequence]


def _diff_gender_token_counters(gender_token_counters: GenderTokenCounters,
                                sort: Optional[bool] = False,
                                limit: Optional[int] = 10,
                                remove_swords: Optional[bool] = False) -> GenderTokenResponse:
    """
    A private helper function that determines the difference of token occurrences
    across multiple Genders.

    :param gender_token_counters: Dict[str, Counter]
    :param sort: Optional[bool], return Dict[str, Sequence[Tuple[str, int]]]
    :param limit: Optional[int], if sort=True, return n=limit number of items in descending order
    :param remove_swords: Optional[bool], remove stop words from return

    >>> token_frequency_1 = Counter({'foo': 1, 'bar': 2, 'own': 4})
    >>> token_frequency_2 = Counter({'foo': 2, 'baz': 3, 'own': 2})
    >>> test = {'Male': token_frequency_1, 'Female': token_frequency_2}
    >>> _diff_gender_token_counters(test).get('Male')
    Counter({'bar': 2, 'own': 2, 'foo': -1})
    >>> _diff_gender_token_counters(test, sort=True).get('Male')
    [('bar', 2), ('own', 2), ('foo', -1)]
    >>> _diff_gender_token_counters(test, sort=True, limit=2).get('Male')
    [('bar', 2), ('own', 2)]
    >>> _diff_gender_token_counters(test, remove_swords=True).get('Male')
    Counter({'bar': 2, 'foo': -1})
    >>> _diff_gender_token_counters(test, sort=True, limit=2, remove_swords=True).get('Male')
    [('bar', 2), ('foo', -1)]
    """

    if not isinstance(limit, int):
        raise ValueError('limit must be of type int')

    difference_dict = {}

    for gender in gender_token_counters:
        current_difference = Counter()

        for word, count in gender_token_counters[gender].items():
            current_difference[word] = count

        for other_gender in gender_token_counters:
            if other_gender == gender:
                continue
            other_adjective_frequency = gender_token_counters[other_gender]

            for word, count in other_adjective_frequency.items():
                if word in current_difference.keys():
                    current_difference[word] -= count

        difference_dict[gender] = current_difference

    if sort:
        return _sort_gender_token_counters(difference_dict,
                                           limit=limit,
                                           remove_swords=remove_swords)
    else:
        if remove_swords:
            for gender in difference_dict:
                for word in list(difference_dict[gender]):
                    if word in SWORDS_ENG:
                        del difference_dict[gender][word]
        return difference_dict


def _generate_token_counter(document: Document,
                            gender_to_find: Gender,
                            word_window: int,
                            tokens: Sequence[str],
                            genders_to_exclude: Optional[Sequence[Gender]] = None) -> Counter:
    # pylint: disable=too-many-locals
    """
    A private helper function for generating token Counters based on the tokenized text of the
    input Document.

    :param document: an instance of the Document class.
    :param gender_to_find: an instance of the Gender class.
    :param word_window: number of words to search for in either direction of a Gender instance.
    :param tokens: a list containing NLTK token strings.
    :param genders_to_exclude: a list containing instances of the Gender class.
    :return: a Counter instance pairing token words with occurrences.

    >>> from gender_analysis.common import MALE, FEMALE
    >>> from corpus_analysis.corpus import Corpus
    >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
    >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
    >>> doc = corpus.documents[-1]
    >>> test = _generate_token_counter(doc, FEMALE, 5, ['NN'], genders_to_exclude=[MALE])
    >>> type(test)
    <class 'collections.Counter'>
    >>> test.get('everyone')
    3
    """

    output = Counter()
    identifiers_to_exclude = []
    text = document.get_tokenized_text()

    identifiers_to_find = gender_to_find.identifiers

    if genders_to_exclude is None:
        genders_to_exclude = list()

    for gender in genders_to_exclude:
        for identifier in gender.identifiers:
            identifiers_to_exclude.append(identifier)

    for words in windowed(text, 2 * word_window + 1):
        if not words[word_window].lower() in identifiers_to_find:
            continue
        if bool(set(words) & set(identifiers_to_exclude)):
            continue

        words = list(words)
        for index, word in enumerate(words):
            words[index] = word.lower()

        tags = nltk.pos_tag(words)
        for tag_index, _ in enumerate(tags):
            if tags[tag_index][1] in tokens:
                word = words[tag_index]
                output[word] += 1

    return output


def _generate_gender_token_counters(document: Document,
                                    genders: Sequence[Gender],
                                    tokens: Sequence[str],
                                    word_window: int) -> GenderTokenCounters:
    """
    A private helper function for generating  dictionaries of the
    shape Dict[str, Counter] that are used throughout this module.

    :param document: an instance of the Document class.
    :param genders: a list containing instances of the Gender class.
    :param tokens: a list containing NLTK token strings.
    :param word_window: number of words to search for in either direction of a Gender instance.
    :return: Dict[str, Counter], with top-level keys being Gender.label.

    >>> from gender_analysis.common import BINARY_GROUP
    >>> from corpus_analysis.corpus import Corpus
    >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
    >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
    >>> doc = corpus.documents[-1]
    >>> test_1 = _generate_gender_token_counters(doc, BINARY_GROUP, ['NN'], 5)
    >>> test_2 = _generate_gender_token_counters(doc, BINARY_GROUP, ['NN'], 10)
    >>> list(test_1.keys()) == ['Female', 'Male']
    True
    >>> test_1.get('Female').get('everyone')
    3
    >>> test_2.get('Female').get('everyone')
    2
    """

    results = {}

    for gender in genders:
        if gender.label == FEMALE.label:
            novel_result = _generate_token_counter(document,
                                                   FEMALE,
                                                   word_window,
                                                   tokens,
                                                   genders_to_exclude=[MALE])
        elif gender.label == MALE.label:
            novel_result = _generate_token_counter(document,
                                                   MALE,
                                                   word_window,
                                                   tokens,
                                                   genders_to_exclude=[FEMALE])
        else:
            # Note that we exclude male from female and female from male but do not do this
            # with other genders.
            novel_result = _generate_token_counter(document, gender, word_window, tokens)
        if novel_result != "lower window bound less than 5":
            results.update({gender.label: novel_result})

    return results


def _merge_token_counters(token_counters: Sequence[Counter]) -> Counter:
    """
    A private helper function for combining multiple dictionaries of the shape token_frequency
    into a single token_frequency.

    :param token_frequencies: a list of the shape [{str: int, ...}, ...]
    :return: a dictionary of the same shape as the above, with all key: value pairs merged.

    >>> test_1 = Counter({'good': 1, 'bad': 1, 'ugly': 1})
    >>> test_2 = Counter({'good': 3, 'bad': 0, 'weird': 2})
    >>> test_3 = Counter({'good': 2, 'bad': 4, 'weird': 0, 'ugly': 2})
    >>> merged_token_frequency = _merge_token_counters([test_1, test_2, test_3])
    >>> merged_token_frequency
    Counter({'good': 6, 'bad': 5, 'ugly': 3, 'weird': 2})
    """

    return reduce(lambda token_counter, total: token_counter + total, token_counters)


def _sort_gender_token_counters(gender_token_counters: GenderTokenCounters,
                                limit: Optional[int] = 10,
                                remove_swords: Optional[bool] = False) -> GenderTokenCounters:
    """
    A private helper function for transforming a dictionary of token instances keyed by
    Gender.label into a sorted list of tuples.

    :param gender_token_counters: Dict[str, Counter].
    :param limit: Optional[int], if sort=True, return n=limit number of items in descending order
    :param remove_swords: Optional[bool], remove stop words from return
    :return: Dict[str, Sequence[Tuple[str, int]]]

    >>> token_frequency_1 = Counter({'foo': 1, 'bar': 2, 'bat': 4, 'own': 3})
    >>> token_frequency_2 = Counter({'foo': 2, 'baz': 3})
    >>> test = {'Male': token_frequency_1, 'Female': token_frequency_2}
    >>> _sort_gender_token_counters(test).get('Male')
    [('bat', 4), ('own', 3), ('bar', 2), ('foo', 1)]
    >>> _sort_gender_token_counters(test).get('Female')
    [('baz', 3), ('foo', 2)]
    >>> _sort_gender_token_counters(test, limit=1).get('Male')
    [('bat', 4)]
    >>> _sort_gender_token_counters(test, remove_swords=True).get('Male')
    [('bat', 4), ('bar', 2), ('foo', 1)]
    >>> _sort_gender_token_counters(test, limit=2, remove_swords=True).get('Male')
    [('bat', 4), ('bar', 2)]
    """
    output_gender_token_counters = {}
    for gender, token_counter in gender_token_counters.items():
        output_gender_token_counters[gender] = {}
        output_gender_token_counters[gender] = _sort_token_counter(token_counter,
                                                                   limit=limit,
                                                                   remove_swords=remove_swords)
    return output_gender_token_counters


def _sort_token_counter(token_counter: Counter,
                        limit: Optional[int] = None,
                        remove_swords: Optional[bool] = False) -> Counter:
    """
    A private helper function for grabbing the most common occurrences in a Counter
    with stop words optionally removed.

    :param limit: Optional[int], if sort=True, return n=limit number of items in descending order
    :param remove_swords: Optional[bool], remove stop words from return
    :return: Sequence[Tuple[str, int]]

    >>> example = Counter({'foo': 1, 'bar': 3, 'bat': 4, 'own': 5})
    >>> _sort_token_counter(example)
    [('own', 5), ('bat', 4), ('bar', 3), ('foo', 1)]
    >>> _sort_token_counter(example, limit=2)
    [('own', 5), ('bat', 4)]
    >>> _sort_token_counter(example, remove_swords=True)
    [('bat', 4), ('bar', 3), ('foo', 1)]
    >>> _sort_token_counter(example, limit=2, remove_swords=True)
    [('bat', 4), ('bar', 3)]
    """
    output_token_counter = token_counter.copy()

    if remove_swords:
        for word in list(output_token_counter):
            if word in SWORDS_ENG:
                del output_token_counter[word]

    return output_token_counter.most_common(limit)


class GenderTokenAnalyzer(Analyzer):
    """
    # TODO: update this and other doctstrings as needed
    The GenderTokenAnalysis instance is a dictionary of the shape
    {Document: {Gender.label: {str(token): int, ...}, ...}, ...},
    with custom helper methods for analyzing occurrences of specific words in a window around
    gendered pronouns.
    """

    def __init__(self,
                 tokens: Optional[Sequence[str]] = None,
                 genders: Optional[Sequence[Gender]] = None,
                 word_window: int = 5,
                 **kwargs) -> None:
        """
        Initializes a GenderTokenAnalyzer object that can be used for retrieving
        analyses concerning the number of occurrences of specific words within a window of
        gendered pronouns.

        :param texts: a Corpus, Document, or list of Documents.
        :param tokens: a list of NLTK token strings, defaulting to adjectives.
        :param genders: a list of Gender instances.
        :param word_window: number of words to search for in either direction of a Gender instance.
        """

        super().__init__(**kwargs)

        if genders is None:
            genders = BINARY_GROUP

        if tokens is None:
            tokens = NLTK_ADJECTIVE_TOKENS

        if not all(isinstance(item, str) for item in tokens):
            raise ValueError('all items in list tokens must be of type str')

        if not all(isinstance(item, Gender) for item in genders):
            raise ValueError('all items in list genders must be of type Gender')

        self.genders = genders
        self.gender_labels = [gender.label for gender in genders]
        self.tokens = tokens
        self.word_window = word_window

        self._by_date = None
        self._by_differences = None
        self._by_document = None
        self._by_gender = None
        self._by_metadata = None
        self._by_overlap = None
        self._by_sorted = None

        results = {}

        for document in self.corpus:
            results[document] = _generate_gender_token_counters(document,
                                                                genders,
                                                                tokens,
                                                                word_window=word_window)

        self._results = results

    def by_date(self,
                time_frame: Tuple[int, int],
                bin_size: int,
                sort: Optional[bool] = False,
                diff: Optional[bool] = False,
                limit: Optional[int] = 10,
                remove_swords: Optional[bool] = False) -> Dict[int, GenderTokenResponse]:
        """
        Return analysis organized by date (as determined by Document metadata).

        :param time_frame: a tuple of the format (start_date, end_date).
        :param bin_size: int for the number of years represented in each list of frequencies
        :param sort: Optional[bool], return Dict[int, Sequence[Tuple[str, int]]]
        :param diff: return the differences between genders.
        :param limit: Optional[int], if sort=True, return n=limit number of items in descending order
        :param remove_swords: Optional[bool], remove stop words from return
        :return: a dictionary of the shape { str(Gender.label): { str(token): int } } or
                 { str(Gender.label): [(str(token), int)] }
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderTokenAnalyzer(file_path=DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer.by_date((2000, 2008), 2).keys()
        dict_keys([2000, 2002, 2004, 2006])
        >>> analyzer.by_date((2000, 2010), 2).get(2002)
        {'Female': Counter({'sad': 7, 'died': 1}), 'Male': Counter()}
        >>> analyzer.by_date((2000, 2010), 2, sort=True).get(2002)
        {'Female': [('sad', 7), ('died', 1)], 'Male': []}
        >>> analyzer.by_date((2000, 2010), 2, diff=True).get(2002)
        {'Female': Counter({'sad': 7, 'died': 1}), 'Male': Counter()}
        """

        hashed_arguments = ''.join([
            str(time_frame[0]),
            str(time_frame[1]),
            str(bin_size),
            str(sort),
            str(diff),
            str(limit),
            str(remove_swords)
        ])

        if self._by_date is None:
            self._by_date = {}
            self._by_date[hashed_arguments] = None
        elif hashed_arguments in self._by_date:
            return self._by_date[hashed_arguments]

        data = {}

        for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
            data[bin_start_year] = {label: Counter() for label in self.gender_labels}

        for document in self.corpus:
            date = getattr(document, 'date', None)
            if date is None:
                continue
            bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
            if bin_year not in data:
                continue
            for gender_label in self.gender_labels:
                data[bin_year][gender_label] = _merge_token_counters(
                    [self._results[document][gender_label], data[bin_year][gender_label]]
                )

        if diff:
            for date, gender_token_frequencies in data.items():
                data[date] = _diff_gender_token_counters(gender_token_frequencies,
                                                         limit=limit,
                                                         sort=sort,
                                                         remove_swords=remove_swords)
        elif sort:
            for date, gender_token_frequencies in data.items():
                data[date] = _sort_gender_token_counters(gender_token_frequencies,
                                                         limit=limit,
                                                         remove_swords=remove_swords)

        self._by_date[hashed_arguments] = data
        return self._by_date[hashed_arguments]

    def by_document(self,
                    sort: Optional[bool] = False,
                    diff: Optional[bool] = False,
                    limit: Optional[int] = 10,
                    remove_swords: Optional[bool] = False) -> Dict[Document, GenderTokenResponse]:

        """
        Return analysis organized by Document.

        :param sort: Optional[bool], return Dict[int, Sequence[Tuple[str, int]]]
        :param diff: return the differences between genders.
        :param limit: Optional[int], if sort=True, return n=limit number of items in descending order
        :param remove_swords: Optional[bool], remove stop words from return
        :return: a dictionary of the shape { str(Gender.label): { str(token): int } } or
                 { str(Gender.label): [(str(token), int)] }
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderTokenAnalyzer(file_path=DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> doc = analyzer.corpus.documents[7]
        >>> list(analyzer.by_document().keys()) == analyzer.corpus.documents
        True
        >>> analyzer.by_document().get(doc)
        {'Female': Counter({'sad': 6, 'died': 1}), 'Male': Counter()}
        """

        hashed_arguments = ''.join([
            str(sort),
            str(diff),
            str(limit),
            str(remove_swords)
        ])

        if self._by_document is None:
            self._by_document = {}
            self._by_document[hashed_arguments] = None
        elif hashed_arguments in self._by_document:
            return self._by_document[hashed_arguments]

        output = {}

        if diff:
            for document in self._results:
                output[document] = {}
                output[document] = _diff_gender_token_counters(self._results[document],
                                                               limit=limit,
                                                               sort=sort,
                                                               remove_swords=remove_swords)
        elif sort:
            output = {}
            for document in self._results:
                token_frequencies = self._results[document]
                output[document] = {}
                output[document] = _sort_gender_token_counters(token_frequencies,
                                                               limit=limit,
                                                               remove_swords=remove_swords)
        else:
            output = self._results

        if len(output) == 1:
            sole_document = list(output)[0]
            output = output[sole_document]

        self._by_document[hashed_arguments] = output
        return output

    def by_gender(self,
                  sort: Optional[bool] = False,
                  diff: Optional[bool] = False,
                  limit: Optional[int] = 10,
                  remove_swords: Optional[bool] = False) -> Dict[str, GenderTokenResponse]:
        """
        Merges all adjectives across texts into dictionaries sorted by gender.

        :param sort: Optional[bool], return Dict[str, Sequence[Tuple[str, int]]]
        :param diff: return the differences between genders.
        :param limit: Optional[int], if sort=True, return n=limit number of items in descending order
        :param remove_swords: Optional[bool], remove stop words from return
        :return: a dictionary of the shape {Gender.label: {str: int, ...}, ...}

        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderTokenAnalyzer(file_path=DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer.by_gender().keys()
        dict_keys(['Female', 'Male'])
        >>> analyzer.by_gender().get('Female')
        Counter({'sad': 14, 'beautiful': 3, 'died': 1})
        >>> analyzer.by_gender(sort=True).get('Female')
        [('sad', 14), ('beautiful', 3), ('died', 1)]
        >>> analyzer.by_gender(diff=True).get('Female')
        Counter({'beautiful': 3, 'died': 1, 'sad': 0})
        >>> analyzer.by_gender(diff=True, sort=True).get('Female')
        [('beautiful', 3), ('died', 1), ('sad', 0)]
        """

        hashed_arguments = f"{str(sort)}{str(diff)}{str(limit)}{remove_swords}"

        if self._by_gender is None:
            self._by_gender = {}
            self._by_gender[hashed_arguments] = None
        elif hashed_arguments in self._by_gender:
            return self._by_gender[hashed_arguments]

        merged_results = {}
        for gender_label in self.gender_labels:
            new_gender_token_counters = [
                self._results[document][gender_label] for document in self._results
            ]
            merged_results[gender_label] = {}
            merged_results[gender_label] = _merge_token_counters(new_gender_token_counters)

        output = merged_results

        if diff:
            output = _diff_gender_token_counters(output,
                                                 limit=limit,
                                                 sort=sort,
                                                 remove_swords=remove_swords)
        elif sort:
            output = _sort_gender_token_counters(output,
                                                 limit=limit,
                                                 remove_swords=remove_swords)

        self._by_gender[hashed_arguments] = output
        return self._by_gender[hashed_arguments]

    def by_metadata(self,
                    metadata_key: str,
                    sort: Optional[bool] = False,
                    diff: Optional[bool] = False,
                    limit: Optional[int] = 10,
                    remove_swords: Optional[bool] = False) -> Dict[str, GenderTokenResponse]:
        """
        Merges all adjectives across texts into dictionaries sorted by gender.

        :param metadata_key: a string.
        :param sort: Optional[bool], return Dict[str, Sequence[Tuple[str, int]]]
        :param diff: return the differences between genders.
        :param limit: Optional[int], if sort=True, return n=limit number of items in descending order
        :param remove_swords: Optional[bool], remove stop words from return
        :return: a dictionary of the shape {Gender.label: {str: int , ...}, ...}.

        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderTokenAnalyzer(file_path=DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer.by_metadata('author_gender').keys()
        dict_keys(['male', 'female'])
        >>> analyzer.by_metadata('author_gender').get('female')
        {'Female': Counter({'sad': 7}), 'Male': Counter({'sad': 12, 'deep': 1})}
        >>> analyzer.by_metadata('author_gender', sort=True).get('female')
        {'Female': [('sad', 7)], 'Male': [('sad', 12), ('deep', 1)]}
        >>> analyzer.by_metadata('author_gender', diff=True).get('female')
        {'Female': Counter({'sad': -5}), 'Male': Counter({'sad': 5, 'deep': 1})}
        """

        hashed_arguments = f"{str(metadata_key)}{str(sort)}{str(diff)}{str(limit)}{remove_swords}"

        if self._by_metadata is None:
            self._by_metadata = {}
            self._by_metadata[hashed_arguments] = None
        elif hashed_arguments in self._by_metadata:
            return self._by_metadata[hashed_arguments]

        data = {}

        for document in self.corpus:
            metadata_attribute = getattr(document, metadata_key, None)
            if metadata_attribute is None:
                continue

            if metadata_attribute not in data:
                data[metadata_attribute] = {}

            for gender_label in self.gender_labels:
                if gender_label not in data[metadata_attribute]:
                    data[metadata_attribute][gender_label] = []
                data[metadata_attribute][gender_label].append(self._results[document][gender_label])

        for key in data:
            for gender_label in self.gender_labels:
                data[key][gender_label] = _merge_token_counters(data[key][gender_label])
            if diff:
                data[key] = _diff_gender_token_counters(data[key],
                                                        limit=limit,
                                                        sort=sort,
                                                        remove_swords=remove_swords)
            elif sort:
                data[key] = _sort_gender_token_counters(data[key],
                                                        limit=limit,
                                                        remove_swords=remove_swords)

        self._by_metadata[hashed_arguments] = data
        return self._by_metadata[hashed_arguments]

    def by_overlap(self) -> Dict[str, Sequence[int]]:
        """
        Looks through the gendered adjectives across the corpus and extracts adjectives that overlap
        across all genders and their occurrences sorted.

        :return: {str: [gender1, gender2, ...], ...}

        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderTokenAnalyzer(file_path=DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer.by_overlap()
        {'sad': [14, 14]}
        """

        if self._by_overlap is not None:
            return self._by_overlap

        overlap_results = {}
        sets_of_adjectives = {}

        for gender_label in self.gender_labels:
            sets_of_adjectives[gender_label] = set(list(self.by_gender()[gender_label].keys()))

        intersects_with_all = set.intersection(*sets_of_adjectives.values())

        for adj in intersects_with_all:
            output = []
            for gender_label in self.gender_labels:
                output.append(self.by_gender()[gender_label][adj])
            overlap_results[adj] = output

        self._by_overlap = overlap_results
        return self._by_overlap

    def get_document(self,
                     metadata_field_key,
                     metadata_field_value,
                     sort: Optional[bool] = False,
                     diff: Optional[bool] = False,
                     limit: Optional[int] = 10,
                     remove_swords: Optional[bool] = False) -> GenderTokenResponse:

        """
        Retrieve a specific Document's analysis from the GenderTokenAnalyzer results,
        with optional formatting arguments.

        :param metadata_field_key: a string.
        :param metadata_field_value: a string.
        :param sort: Optional[bool], return Dict[str, Sequence[Tuple[str, int]]]
        :param diff: return the differences between genders.
        :param limit: Optional[int], if sort=True, return n=limit number of items in descending order
        :param remove_swords: Optional[bool], remove stop words from return
        :return: a dictionary of the shape {Gender.label: {str: int , ...}, ...}.

        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> analyzer = GenderTokenAnalyzer(file_path=DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer.get_document('title', 'Title 7')
        {'Female': Counter(), 'Male': Counter({'handsome': 3, 'sad': 1})}
        """

        document = self.corpus.get_document(metadata_field_key, metadata_field_value)

        if document is not None:
            gender_token_counters = self.by_document().get(document)
            if gender_token_counters is not None:
                if diff:
                    return _diff_gender_token_counters(gender_token_counters,
                                                       limit=limit,
                                                       sort=sort,
                                                       remove_swords=remove_swords)
                elif sort:
                    return _sort_gender_token_counters(gender_token_counters,
                                                       limit=limit,
                                                       remove_swords=remove_swords)
                else:
                    return gender_token_counters

        return None

    def store(self, pickle_filepath: Optional[str] = 'gender_tokens_analysis.pgz') -> None:
        """
        Saves self to a pickle file.

        :param pickle_filepath: filepath to save the output.
        :return: None, saves results as pickled file with name 'gender_tokens_analysis'
        """

        try:
            load_pickle(pickle_filepath)
            user_inp = input("results already stored. overwrite previous analysis? (y/n)")
            if user_inp == 'y':
                store_pickle(self, pickle_filepath)
            else:
                pass
        except IOError:
            store_pickle(self, pickle_filepath)
