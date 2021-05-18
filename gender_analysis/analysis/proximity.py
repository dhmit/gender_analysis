from collections import Counter
from typing import Dict, Optional, Sequence, Tuple, Union
from functools import reduce
from more_itertools import windowed
import nltk

#from corpus_analysis.corpus import Corpus
from corpus_analysis.document import Document
from corpus_analysis.common import load_pickle, store_pickle, NLTK_TAGS, NLTK_TAGS_ADJECTIVES

from gender_analysis.common import MALE, FEMALE, BINARY_GROUP, SWORDS_ENG
from gender_analysis.gender import Gender

GenderTokenCounters = Dict[str, Counter]
GenderTokenSequence = Dict[str, Sequence[Tuple[str, int]]]
GenderTokenResponse = Union[GenderTokenCounters, GenderTokenSequence]


def _diff_gender_token_counters(gender_token_counters: GenderTokenCounters,
                                sort: bool = False,
                                limit: int = 10,
                                remove_swords: bool = False) -> GenderTokenResponse:
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


def _generate_token_counter(document,
                            gender_to_find: Gender,
                            word_window: int,
                            tags: Sequence[str],
                            genders_to_exclude: Optional[Sequence[Gender]] = None) -> Counter:
    # pylint: disable=too-many-locals
    """
    A private helper function for generating token Counters based on the tokenized text of the
    input Document.

    :param document: an instance of the Document class.
    :param gender_to_find: an instance of the Gender class.
    :param word_window: number of words to search for in either direction of a Gender instance.
    :param tags: a list containing NLTK token strings.
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
        if words[word_window].lower() in identifiers_to_find:
            if bool(set(words) & set(identifiers_to_exclude)):
                continue

            words = list(words)
            for index, word in enumerate(words):
                words[index] = word.lower()

            tagged_tokens = nltk.pos_tag(words)
            for tag_index, _ in enumerate(tagged_tokens):
                if tagged_tokens[tag_index][1] in tags:
                    word = words[tag_index]
                    output[word] += 1

    return output


def _generate_gender_token_counters(document,
                                    genders: Sequence[Gender],
                                    tags: Sequence[str],
                                    word_window: int) -> GenderTokenCounters:
    """
    A private helper function for generating  dictionaries of the
    shape Dict[str, Counter] that are used throughout this module.

    :param document: an instance of the Document class.
    :param genders: a list containing instances of the Gender class.
    :param tags: a list containing NLTK token strings.
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
                                                   tags,
                                                   genders_to_exclude=[MALE])
        elif gender.label == MALE.label:
            novel_result = _generate_token_counter(document,
                                                   MALE,
                                                   word_window,
                                                   tags,
                                                   genders_to_exclude=[FEMALE])
        else:
            # Note that we exclude male from female and female from male but do not do this
            # with other genders.
            novel_result = _generate_token_counter(document, gender, word_window, tags)
        if novel_result != "lower window bound less than 5":
            results.update({gender.label: novel_result})

    return results


def _merge_token_counters(token_counters: Sequence[Counter]) -> Counter:
    """
    A private helper function for combining multiple dictionaries of the shape token_frequency
    into a single token_frequency.

    :param token_counters: a list of the shape [{str: int, ...}, ...]
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
                                limit: int = 10,
                                remove_swords: bool = False) -> GenderTokenSequence:
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
                        remove_swords: bool = False) -> Sequence[Tuple[str, int]]:
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


def find_in_document_gender(document,
                            gender: Gender,
                            tags: Sequence[str] = None,
                            word_window: int = 5,
                            genders_to_exclude: Optional[Sequence[Gender]] = None) -> Counter:
    """
    Returns a Counter of words and occurrences found within a window of gender identifiers.

    :param document: an instance of the Document class.
    :param gender: an instance of the Gender class.
    :param tags: a list containing NLTK token strings.
    :param word_window: number of words to search for in either direction of a Gender instance.
    :param genders_to_exclude: a list containing instances of the Gender class.
    :return: a Counter instance pairing token words with occurrences.

    >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
    >>> from corpus_analysis import corpus
    >>> from gender_analysis import common
    >>> corpus = corpus.Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
    >>> analyzer = GenderProximityAnalyzer(corpus)
    >>> doc = analyzer.documents[6]
    >>> find_in_document_gender(doc, common.FEMALE, genders_to_exclude=[common.MALE])
    Counter({'sad': 1})
    """

    if tags is None:
        tags = NLTK_TAGS_ADJECTIVES

    return _generate_token_counter(document,
                                   gender,
                                   word_window,
                                   tags,
                                   genders_to_exclude=genders_to_exclude)


def find_in_document_female(document,
                            tags: Sequence[str] = None,
                            word_window: int = 5) -> Counter:
    """
    Returns a Counter of words and occurrences found within a window of FEMALE identifiers.

    :param document: an instance of the Document class.
    :param word_window: number of words to search for in either direction of a Gender instance.
    :param tags: a list containing NLTK token strings.
    :return: a Counter instance pairing token words with occurrences.

    >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
    >>> from corpus_analysis import corpus
    >>> corpus = corpus.Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
    >>> analyzer = GenderProximityAnalyzer(corpus)
    >>> doc = analyzer.documents[6]
    >>> find_in_document_female(doc)
    Counter({'sad': 1})
    """

    if tags is None:
        tags = NLTK_TAGS_ADJECTIVES

    return _generate_token_counter(document,
                                   FEMALE,
                                   word_window,
                                   tags,
                                   genders_to_exclude=[MALE])


def find_in_document_male(document: Document,
                          tags: Sequence[str] = None,
                          word_window: int = 5) -> Counter:
    """
    Returns a Counter of words and occurrences found within a window of MALE identifiers.

    :param document: an instance of the Document class.
    :param word_window: number of words to search for in either direction of a Gender instance.
    :param tags: a list containing NLTK token strings.
    :return: a Counter instance pairing token words with occurrences.

    >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
    >>> from corpus_analysis import corpus
    >>> corpus = corpus.Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
    >>> analyzer = GenderProximityAnalyzer(corpus)
    >>> doc = analyzer.documents[3]
    >>> find_in_document_male(doc)
    Counter({'deep': 1})
    """

    if tags is None:
        tags = NLTK_TAGS_ADJECTIVES

    return _generate_token_counter(document,
                                   MALE,
                                   word_window,
                                   tags,
                                   genders_to_exclude=[FEMALE])


class GenderProximityAnalyzer:
    """
    The GenderProximityAnalyzer instance finds word occurrences within a window around
    gendered pronouns. Helper methods are provided to organize and analyze those occurrences
    according to relevant criteria.

    Class methods:
        list_nltk_tags()

    Instance methods:
        by_date()
        by_document()
        by_gender()
        by_metadata()
        by_overlap()
    """

    def __init__(self,
                 texts,
                 tags: Optional[Sequence[str]] = None,
                 genders: Optional[Sequence[Gender]] = None,
                 word_window: int = 5) -> None:
        """
        Initializes a GenderProximityAnalyzer object that can be used for retrieving
        analyses concerning the number of occurrences of specific words within a window of
        gendered pronouns.

        :param texts: a Corpus, Document, or list of Documents.
        :param tags: a list of NLTK token strings, defaulting to adjectives.
        :param genders: a list of Gender instances.
        :param word_window: number of words to search for in either direction of a Gender instance.
        """

        if genders is None:
            genders = BINARY_GROUP

        if tags is None:
            tags = NLTK_TAGS_ADJECTIVES

        #if isinstance(texts, Corpus):
            #documents = texts.documents
        elif isinstance(texts, Document):
            documents = [texts]
        elif isinstance(texts, list):
            if all(isinstance(item, Document) for item in texts):
                documents = texts
            else:
                raise ValueError('all items in list texts must be of type Document')
        else:
            raise ValueError('texts must be of type Document, Corpus, or a list of Documents')

        if not all(isinstance(item, str) for item in tags):
            raise ValueError('all items in list tags must be of type str')

        if not all(isinstance(item, Gender) for item in genders):
            raise ValueError('all items in list genders must be of type Gender')

        self.documents = documents
        self.genders = genders
        self.gender_labels = [gender.label for gender in genders]
        self.tags = tags
        self.word_window = word_window

        # cached results
        self._by_gender = None

        results = {}

        for document in self.documents:
            results[document] = _generate_gender_token_counters(document,
                                                                genders,
                                                                tags,
                                                                word_window=word_window)

        self._results = results

    @classmethod
    def list_nltk_tags(cls) -> None:
        """
        Print out possible NLTK tags to use when initializing a new GenderProximityAnalyzer.

        :return: None
        """

        for tag, definition in NLTK_TAGS.items():
            print(f'{tag}: {definition}')

    def by_date(self,
                time_frame: Tuple[int, int],
                bin_size: int,
                sort: bool = False,
                diff: bool = False,
                limit: int = 10,
                remove_swords: bool = False) -> Dict[int, GenderTokenResponse]:
        """
        Return analysis organized by date (as determined by Document metadata).

        :param time_frame: a tuple of the format (start_date, end_date).
        :param bin_size: int for the number of years represented in each list of frequencies
        :param sort: Optional[bool], return Dict[int, Sequence[Tuple[str, int]]]
        :param diff: return the differences between genders.
        :param limit: Optional[int], if sort=True, return n=limit number of items in desc order.
        :param remove_swords: Optional[bool], remove stop words from return
        :return: a dictionary of the shape { str(Gender.label): { str(token): int } } or
                 { str(Gender.label): [(str(token), int)] }
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> from corpus_analysis import corpus
        >>> corpus = corpus.Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer = GenderProximityAnalyzer(corpus)
        >>> analyzer.by_date((2000, 2008), 2).keys()
        dict_keys([2000, 2002, 2004, 2006])
        >>> analyzer.by_date((2000, 2010), 2).get(2002)
        {'Female': Counter({'sad': 7, 'died': 1}), 'Male': Counter()}
        >>> analyzer.by_date((2000, 2010), 2, sort=True).get(2002)
        {'Female': [('sad', 7), ('died', 1)], 'Male': []}
        >>> analyzer.by_date((2000, 2010), 2, diff=True).get(2002)
        {'Female': Counter({'sad': 7, 'died': 1}), 'Male': Counter()}
        """

        output = {}

        for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
            output[bin_start_year] = {label: Counter() for label in self.gender_labels}

        for document in self.documents:
            date = getattr(document, 'date', None)
            if date is None:
                continue
            bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
            if bin_year not in output:
                continue
            for gender_label in self.gender_labels:
                output[bin_year][gender_label] = _merge_token_counters(
                    [self._results[document][gender_label], output[bin_year][gender_label]]
                )

        if diff:
            for date, gender_token_frequencies in output.items():
                output[date] = _diff_gender_token_counters(gender_token_frequencies,
                                                           limit=limit,
                                                           sort=sort,
                                                           remove_swords=remove_swords)
        elif sort:
            for date, gender_token_frequencies in output.items():
                output[date] = _sort_gender_token_counters(gender_token_frequencies,
                                                           limit=limit,
                                                           remove_swords=remove_swords)

        return output

    def by_document(self,
                    sort: bool = False,
                    diff: bool = False,
                    limit: int = 10,
                    remove_swords: bool = False) -> Dict[str, GenderTokenResponse]:

        """
        Return analysis organized by Document.

        :param sort: Optional[bool], return Dict[int, Sequence[Tuple[str, int]]]
        :param diff: return the differences between genders.
        :param limit: Optional[int], if sort=True, return n=limit number of items in desc order.
        :param remove_swords: Optional[bool], remove stop words from return
        :return: a dictionary of the shape { str(Gender.label): { str(token): int } } or
                 { str(Gender.label): [(str(token), int)] }
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> from corpus_analysis import corpus
        >>> corpus = corpus.Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer = GenderProximityAnalyzer(corpus)
        >>> doc = analyzer.documents[7]
        >>> analyzer_document_labels = list(analyzer.by_document().keys())
        >>> document_labels = list(map(lambda d: d.label, analyzer.documents))
        >>> analyzer_document_labels == document_labels
        True
        >>> analyzer.by_document().get(doc.label)
        {'Female': Counter({'sad': 6, 'died': 1}), 'Male': Counter()}
        """

        output = {}

        if diff:
            for document in self._results:
                output[document.label] = _diff_gender_token_counters(self._results[document],
                                                                     limit=limit,
                                                                     sort=sort,
                                                                     remove_swords=remove_swords)
        elif sort:
            for document in self._results:
                token_frequencies = self._results[document]
                output[document.label] = _sort_gender_token_counters(token_frequencies,
                                                                     limit=limit,
                                                                     remove_swords=remove_swords)
        else:
            for document in self._results:
                output[document.label] = self._results[document]

        return output

    def by_gender(self,
                  sort: bool = False,
                  diff: bool = False,
                  limit: int = 10,
                  remove_swords: bool = False) -> Dict[str, GenderTokenResponse]:
        """
        Return analysis organized by Document. Merges all words across texts
        into dictionaries sorted by gender.

        :param sort: Optional[bool], return Dict[str, Sequence[Tuple[str, int]]]
        :param diff: return the differences between genders.
        :param limit: Optional[int], if sort=True, return n=limit number of items in desc order.
        :param remove_swords: Optional[bool], remove stop words from return
        :return: a dictionary of the shape {Gender.label: {str: int, ...}, ...}

        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> from corpus_analysis import corpus
        >>> corpus = corpus.Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer = GenderProximityAnalyzer(corpus)
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
            self._by_gender = {hashed_arguments: None}
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
        return output

    def by_metadata(self,
                    metadata_key: str,
                    sort: bool = False,
                    diff: bool = False,
                    limit: int = 10,
                    remove_swords: bool = False) -> Dict[str, GenderTokenResponse]:
        """
        Return analysis organized by Document metadata. Merges all words across texts
        into dictionaries sorted by provided metadata_key.

        :param metadata_key: a string.
        :param sort: Optional[bool], return Dict[str, Sequence[Tuple[str, int]]]
        :param diff: return the differences between genders.
        :param limit: Optional[int], if sort=True, return n=limit number of items in desc order.
        :param remove_swords: Optional[bool], remove stop words from return
        :return: a dictionary of the shape {Gender.label: {str: int , ...}, ...}.

        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> from corpus_analysis import corpus
        >>> corpus = corpus.Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer = GenderProximityAnalyzer(corpus)
        >>> analyzer.by_metadata('author_gender').keys()
        dict_keys(['male', 'female'])
        >>> analyzer.by_metadata('author_gender').get('female')
        {'Female': Counter({'sad': 7}), 'Male': Counter({'sad': 12, 'deep': 1})}
        >>> analyzer.by_metadata('author_gender', sort=True).get('female')
        {'Female': [('sad', 7)], 'Male': [('sad', 12), ('deep', 1)]}
        >>> analyzer.by_metadata('author_gender', diff=True).get('female')
        {'Female': Counter({'sad': -5}), 'Male': Counter({'sad': 5, 'deep': 1})}
        """

        output = {}

        for document in self.documents:
            matching_key = getattr(document, metadata_key, None)
            if matching_key is None:
                continue

            if matching_key not in output:
                output[matching_key] = {gender_label: [] for gender_label in self.gender_labels}

            for gender_label in self.gender_labels:
                output[matching_key][gender_label].append(self._results[document][gender_label])

        for key in output:
            for gender_label in self.gender_labels:
                output[key][gender_label] = _merge_token_counters(output[key][gender_label])
            if diff:
                output[key] = _diff_gender_token_counters(output[key],
                                                          limit=limit,
                                                          sort=sort,
                                                          remove_swords=remove_swords)
            elif sort:
                output[key] = _sort_gender_token_counters(output[key],
                                                          limit=limit,
                                                          remove_swords=remove_swords)

        return output

    def by_overlap(self) -> Dict[str, Sequence[int]]:
        """
        Looks through the gendered words across the corpus and extracts words that overlap
        across all genders and their occurrences sorted.

        :return: {str: [gender1, gender2, ...], ...}

        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> from corpus_analysis import corpus
        >>> corpus = corpus.Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analyzer = GenderProximityAnalyzer(corpus)
        >>> analyzer.by_overlap()
        {'sad': {'Female': 14, 'Male': 14}}
        """

        output = {}
        sets_of_adjectives = {}

        for gender_label in self.gender_labels:
            sets_of_adjectives[gender_label] = set(list(self.by_gender()[gender_label].keys()))

        intersects_with_all = set.intersection(*sets_of_adjectives.values())

        for adj in intersects_with_all:
            results_by_gender = {}
            for gender_label in self.gender_labels:
                results_by_gender[gender_label] = self.by_gender().get(gender_label).get(adj)
            output[adj] = results_by_gender

        return output

    def store(self, pickle_filepath: str = 'gender_proximity_analysis.pgz') -> None:
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
