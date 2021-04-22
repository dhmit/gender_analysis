from collections import Counter, UserDict
from more_itertools import windowed
import nltk

from corpus_analysis.corpus import Corpus
from corpus_analysis.document import Document
from corpus_analysis.common import load_pickle, store_pickle
from gender_analysis.common import MALE, FEMALE, BINARY_GROUP, SWORDS_ENG
from gender_analysis.gender import Gender


NLTK_ADJECTIVE_TOKENS = ["JJ", "JJR", "JJS"]
NLTK_NOUN_TOKENS = ['NN']


def _generate_token_occurrences(document,
                                gender_to_find,
                                word_window,
                                tokens,
                                genders_to_exclude=None):
    # pylint: disable=too-many-locals
    """
    A private helper function for generating token_frequencies, a dictionary of the shape
    {str: int, ...} that is used throughout this module.

    :param document: an instance of the Document class.
    :param gender_to_find: an instance of the Gender class.
    :param word_window: number of words to search for in either direction of a Gender instance.
    :param tokens: a list containing NLTK token strings.
    :param genders_to_exclude: a list containing instances of the Gender class.
    :return: an instance of a dict of the shape {str: int, ...}.

    >>> from gender_analysis.common import MALE, FEMALE
    >>> from corpus_analysis.corpus import Corpus
    >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
    >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
    >>> doc = corpus.documents[-1]
    >>> test = _generate_token_occurrences(doc, FEMALE, 5, ['NN'], genders_to_exclude=[MALE])
    >>> test
    {'jane': 1, 'adultery': 1, 'gal': 2, 'everyone': 3, 'thought': 1, 'everybody': 2, 'corpse': 1}
    """

    output = {}
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
                if word in output.keys():
                    output[word] += 1
                else:
                    output[word] = 1

    return output


def _generate_gender_token_occurrences(document, genders, tokens, word_window):
    """
    A private helper function for generating gender_token_frequencies, a dictionary of the
    shape {Gender.label: {str: int, ...}, ...} that is used throughout this module.

    :param document: an instance of the Document class.
    :param genders: a list containing instances of the Gender class.
    :param tokens: a list containing NLTK token strings.
    :param word_window: number of words to search for in either direction of a Gender instance.
    :return: an instance of a dict of the shape {Gender.label: {str: int, ...}, ...}.

    >>> from gender_analysis.common import BINARY_GROUP
    >>> from corpus_analysis.corpus import Corpus
    >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
    >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
    >>> doc = corpus.documents[-1]
    >>> test_1 = _generate_gender_token_occurrences(doc, BINARY_GROUP, ['NN'], 5)
    >>> test_2 = _generate_gender_token_occurrences(doc, BINARY_GROUP, ['NN'], 10)
    >>> list(test_1.keys()) == ['Female', 'Male']
    True
    >>> test_1.get('Female')
    {'jane': 1, 'adultery': 1, 'gal': 2, 'everyone': 3, 'thought': 1, 'everybody': 2, 'corpse': 1}
    >>> test_2.get('Female')
    {'adultery': 1, 'gal': 1, 'everyone': 2, 'thought': 1, 'everybody': 2, 'corpse': 1}
    """

    results = {}

    for gender in genders:
        if gender.label == FEMALE.label:
            novel_result = _generate_token_occurrences(document,
                                                       FEMALE,
                                                       word_window,
                                                       tokens,
                                                       genders_to_exclude=[MALE])
        elif gender.label == MALE.label:
            novel_result = _generate_token_occurrences(document,
                                                       MALE,
                                                       word_window,
                                                       tokens,
                                                       genders_to_exclude=[FEMALE])
        else:
            # Note that we exclude male from female and female from male but do not do this
            # with other genders.
            novel_result = _generate_token_occurrences(document, gender, word_window, tokens)
        if novel_result != "lower window bound less than 5":
            results.update({gender.label: novel_result})

    return results


def _diff_gender_token_occurrences(gender_token_frequencies,
                                   sort=False,
                                   limit=10,
                                   remove_swords=False):
    """
    A private helper function that determines the difference of token occurrences
    across multiple Genders.

    :param gender_token_frequencies: a dictionary of the shape {Gender.label: { str: int, ...}, ...}
    :param sort: boolean, return a sorted list of the shape [(str: int), ...]
    :param limit: integer, the number of items to return in each list.
    :param remove_swords: boolean, remove stop words.

    >>> token_frequency1 = { 'foo': 1, 'bar': 2, 'own': 4 }
    >>> token_frequency2 = { 'foo': 2, 'baz': 3, 'own': 2 }
    >>> test = { 'Male': token_frequency1, 'Female': token_frequency2 }
    >>> _diff_gender_token_occurrences(test).get('Male')
    {'foo': -1, 'bar': 2, 'own': 2}
    >>> _diff_gender_token_occurrences(test, sort=True).get('Male')
    [('bar', 2), ('own', 2), ('foo', -1)]
    >>> _diff_gender_token_occurrences(test, sort=True, limit=2).get('Male')
    [('bar', 2), ('own', 2)]
    >>> _diff_gender_token_occurrences(test, remove_swords=True).get('Male')
    {'foo': -1, 'bar': 2}
    >>> _diff_gender_token_occurrences(test, sort=True, limit=2, remove_swords=True).get('Male')
    [('bar', 2), ('foo', -1)]
    """

    if not isinstance(limit, int):
        raise ValueError('limit must be of type int')

    difference_dict = {}

    for gender in gender_token_frequencies:
        current_difference = {}

        for word, count in gender_token_frequencies[gender].items():
            current_difference[word] = count

        for other_gender in gender_token_frequencies:
            if other_gender == gender:
                continue
            other_adjective_frequency = gender_token_frequencies[other_gender]

            for word, count in other_adjective_frequency.items():
                if word in current_difference.keys():
                    current_difference[word] -= count

        difference_dict[gender] = current_difference

    if sort:
        return _sort_gender_token_occurrences(difference_dict,
                                              limit=limit,
                                              remove_swords=remove_swords)
    else:
        if remove_swords:
            for gender in difference_dict:
                copy = difference_dict[gender].items()
                new_token_frequencies = {key: count for key, count in copy if key not in SWORDS_ENG}
                difference_dict[gender] = new_token_frequencies
        return difference_dict


def _merge_token_occurrences(token_frequencies):
    """
    A private helper function for combining multiple dictionaries of the shape token_frequency
    into a single token_frequency.

    :param token_frequencies: a list of the shape [{str: int, ...}, ...]
    :return: a dictionary of the same shape as the above, with all key: value pairs merged.

    >>> test_1 = { 'good': 1, 'bad': 1, 'ugly': 1 }
    >>> test_2 = { 'good': 3, 'bad': 0, 'weird': 2 }
    >>> test_3 = { 'good': 2, 'bad': 4, 'weird': 0, 'ugly': 2 }
    >>> merged_token_frequency = _merge_token_occurrences([test_1, test_2, test_3])
    >>> merged_token_frequency
    {'good': 6, 'bad': 5, 'ugly': 3, 'weird': 2}
    """

    merged_token_frequencies = {}
    for token_frequency in token_frequencies:
        for (key, value) in token_frequency.items():
            if key in merged_token_frequencies:
                merged_token_frequencies[key] += value
            else:
                merged_token_frequencies[key] = value
    return merged_token_frequencies


def _sort_token_occurrences(token_frequencies, limit=None, remove_swords=False):
    """
    A private helper function for transforming a dict of type {str: int}
    into a sorted list of length limit and shape [(str, int), ...].

    :param limit: the intenger number of items to return in each list.
    :param remove_swords: boolean, remove stop words.
    :return: a list of the shape [(str, int), ...]

    >>> example = { 'foo': 1, 'bar': 3, 'bat': 4, 'own': 5 }
    >>> _sort_token_occurrences(example)
    [('own', 5), ('bat', 4), ('bar', 3), ('foo', 1)]
    >>> _sort_token_occurrences(example, limit=2)
    [('own', 5), ('bat', 4)]
    >>> _sort_token_occurrences(example, remove_swords=True)
    [('bat', 4), ('bar', 3), ('foo', 1)]
    >>> _sort_token_occurrences(example, limit=2, remove_swords=True)
    [('bat', 4), ('bar', 3)]
    """
    output = token_frequencies.copy()

    if remove_swords:
        output = {key: count for key, count in output.items() if key not in SWORDS_ENG}
    sorted_counter = Counter(output).most_common()

    if limit is None:
        limit = len(sorted_counter)
    return sorted_counter[:limit]


def _sort_gender_token_occurrences(gender_token_frequencies, limit=10, remove_swords=False):
    """
    A private helper function for transforming a dictionary of token instances keyed by
    Gender.label into a sorted list of tuples.
    Accepts a list of dictionaries of the shape {Gender.label: {str: int, ...}, ...}
    and returns a dictionary of the shape {Gender.label: [(str, int), ...], ...}
    with all int values combined and sorted.

    :param gender_token_frequencies: a dictionary of the shape {Gender.label: {str: int, ...}, ...}.
    :param limit: integer, the number of items to return in each list.
    :param remove_swords: boolean, remove stop words.
    :return: a dict of the shape {Gender.label: [(str, int), ...], ...}

    >>> token_frequency_1 = { 'foo': 1, 'bar': 2, 'bat': 4, 'own': 3 }
    >>> token_frequency_2 = { 'foo': 2, 'baz': 3 }
    >>> test = { 'Male': token_frequency_1, 'Female': token_frequency_2 }
    >>> _sort_gender_token_occurrences(test).get('Male')
    [('bat', 4), ('own', 3), ('bar', 2), ('foo', 1)]
    >>> _sort_gender_token_occurrences(test).get('Female')
    [('baz', 3), ('foo', 2)]
    >>> _sort_gender_token_occurrences(test, limit=1).get('Male')
    [('bat', 4)]
    >>> _sort_gender_token_occurrences(test, remove_swords=True).get('Male')
    [('bat', 4), ('bar', 2), ('foo', 1)]
    >>> _sort_gender_token_occurrences(test, limit=2, remove_swords=True).get('Male')
    [('bat', 4), ('bar', 2)]
    """
    output = {}
    for gender, token_frequencies in gender_token_frequencies.items():
        output[gender] = {}
        output[gender] = _sort_token_occurrences(token_frequencies,
                                                 limit=limit,
                                                 remove_swords=remove_swords)
    return output


class GenderTokenAnalysis(UserDict):
    """
    The GenderTokenAnalysis instance is a dictionary of the shape
    {Document: {Gender.label: {str(token): int, ...}, ...}, ...},
    with custom helper methods for analyzing occurrences of specific words in a window around
    gendered pronouns.
    """

    def __init__(self, texts, tokens=None, genders=None, word_window=5):
        """
        Initializes a GenderTokenAnalysis object that can be used for retrieving
        analyses concerning the number of occurrences of specific words within a window of
        gendered pronouns.

        :param texts: a Corpus, Document, or list of Documents.
        :param tokens: a list of NLTK token strings, defaulting to adjectives.
        :param genders: a list of Gender instances.
        :param word_window: number of words to search for in either direction of a Gender instance.

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> document = corpus.documents[-1]
        >>> analysis = GenderTokenAnalysis(corpus)
        >>> list(analysis.keys()) == corpus.documents
        True
        >>> analysis.get(document).get('Female').get('beautiful')
        3
        """

        if genders is None:
            genders = BINARY_GROUP

        if tokens is None:
            tokens = NLTK_ADJECTIVE_TOKENS

        if isinstance(texts, Corpus):
            sanitized_documents = texts.documents
        elif isinstance(texts, Document):
            sanitized_documents = [texts]
        elif isinstance(texts, list):
            if all(isinstance(item, Document) for item in texts):
                sanitized_documents = texts
            else:
                raise ValueError('all items in list texts must be of type Document')
        else:
            raise ValueError('texts must be of type Document, Corpus, or a list of Documents')

        if not isinstance(tokens, list):
            raise ValueError('tokens must be a list of NLTK token strings')

        if not all(isinstance(item, str) for item in tokens):
            raise ValueError('all items in list tokens must be of type str')

        if not all(isinstance(item, Gender) for item in genders):
            raise ValueError('all items in list genders must be of type Gender')

        self.documents = sanitized_documents
        self.genders = genders
        self.gender_labels = [gender.label for gender in genders]
        self.tokens = tokens
        self.word_window = word_window
        self._by_date = None
        self._by_differences = None
        self._by_gender = None
        self._by_metadata = None
        self._by_overlap = None
        self._by_sorted = None

        analysis = {}

        for document in self.documents:
            analysis[document] = _generate_gender_token_occurrences(document,
                                                                    genders,
                                                                    tokens,
                                                                    word_window=word_window)

        super().__init__(analysis)

    def by_date(self, time_frame, bin_size, sort=False, diff=False, limit=10, remove_swords=False):
        """
        Return analysis in the format { int(date): { str(Gender.label: { str(token): int } } }

        :param time_frame: a tuple of the format (start_date, end_date).
        :param bin_size: int for the number of years represented in each list of frequencies
        :param sort: return results in a sorted list.
        :param diff: return the differences between genders.
        :param limit: if sort=True, restrict output to top occurrences sorted.
        :param remove_swords: if sort=True, remove stop words from results.
        :return: a dictionary of the shape { str(Gender.label): { str(token): int } } or
                 { str(Gender.label): [(str(token), int)] }
        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analysis = GenderTokenAnalysis(corpus)
        >>> list(analysis.by_date((2000, 2008), 2).keys()) == [2000, 2002, 2004, 2006]
        True
        >>> analysis.by_date((2000, 2010), 2).get(2002).get('Female').get('sad')
        7
        >>> analysis.by_date((2000, 2010), 2, sort=True).get(2002).get('Female')[0]
        ('sad', 7)
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
            data[bin_start_year] = {label: {} for label in self.gender_labels}

        for document in self.documents:
            date = getattr(document, 'date', None)
            if date is None:
                continue
            bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
            if bin_year not in data:
                continue
            for gender_label in self.gender_labels:
                data[bin_year][gender_label] = _merge_token_occurrences(
                    [self[document][gender_label], data[bin_year][gender_label]]
                )

        if diff:
            for date, gender_token_frequencies in data.items():
                data[date] = _diff_gender_token_occurrences(gender_token_frequencies,
                                                            limit=limit,
                                                            sort=sort,
                                                            remove_swords=remove_swords)
        elif sort:
            for date, gender_token_frequencies in data.items():
                data[date] = _sort_gender_token_occurrences(gender_token_frequencies,
                                                            limit=limit,
                                                            remove_swords=remove_swords)

        self._by_date[hashed_arguments] = data
        return self._by_date[hashed_arguments]

    def by_differences(self, sort=False, limit=10, remove_swords=False):
        """
        Return results indicating the difference of token occurrences
        across multiple Genders.

        :param sort: if True, return differences in a sorted list
        :param limit: if sort=True, restrict output to top occurrences sorted.
        :param remove_swords: if sort=True, remove stop words from results.
        :return: a dictionary of the shape {Gender.label: {str: int, ...}, ...}

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> document = corpus.documents[-1]
        >>> analysis = GenderTokenAnalysis(corpus)
        >>> analysis.by_differences().keys() == analysis.keys()
        True
        >>> analysis.by_differences().get(document).get('Female')
        {'beautiful': 3, 'sad': 1}
        >>> analysis.by_differences(sort=True).get(document).get('Female')
        [('beautiful', 3), ('sad', 1)]
        """

        hashed_arguments = f"{str(sort)}{str(limit)}{remove_swords}"

        if self._by_differences is None:
            self._by_differences = {}
            self._by_differences[hashed_arguments] = None
        elif hashed_arguments in self._by_differences:
            return self._by_differences[hashed_arguments]

        diff = {}

        for document in self:
            diff[document] = {}
            diff[document] = _diff_gender_token_occurrences(self[document],
                                                            limit=limit,
                                                            sort=sort,
                                                            remove_swords=remove_swords)

        self._by_differences[hashed_arguments] = diff
        return self._by_differences[hashed_arguments]

    def by_gender(self, sort=False, diff=False, limit=10, remove_swords=False):
        """
        Merges all adjectives across texts into dictionaries sorted by gender.

        :param sort: return results in a sorted list.
        :param diff: return the differences between genders.
        :param limit: if sort=True, restrict output to top occurrences sorted.
        :param remove_swords: if sort=True, remove stop words from results.
        :return: a dictionary of the shape {Gender.label: {str: int, ...}, ...}

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analysis = GenderTokenAnalysis(corpus)
        >>> list(analysis.by_gender().keys()) == ['Female', 'Male']
        True
        >>> analysis.by_gender().get('Female').get('beautiful')
        3
        >>> analysis.by_gender().get('Male').get('handsome')
        6
        >>> analysis.by_gender(sort=True).get('Male')[0]
        ('sad', 14)
        >>> analysis.by_gender(diff=True).get('Male')
        {'deep': 2, 'sad': 0, 'handsome': 6}
        >>> analysis.by_gender(diff=True, sort=True).get('Male')
        [('handsome', 6), ('deep', 2), ('sad', 0)]
        >>> analysis.by_gender(diff=True, sort=True, limit=2).get('Male')
        [('handsome', 6), ('deep', 2)]
        """

        hashed_arguments = f"{str(sort)}{str(diff)}{str(limit)}{remove_swords}"

        if self._by_gender is None:
            self._by_gender = {}
            self._by_gender[hashed_arguments] = None
        elif hashed_arguments in self._by_gender:
            return self._by_gender[hashed_arguments]

        merged_results = {}
        for gender_label in self.gender_labels:
            new_gender_token_frequencies = [self[document][gender_label] for document in self]
            merged_results[gender_label] = {}
            merged_results[gender_label] = _merge_token_occurrences(new_gender_token_frequencies)

        output = merged_results

        if diff:
            output = _diff_gender_token_occurrences(output,
                                                    limit=limit,
                                                    sort=sort,
                                                    remove_swords=remove_swords)
        elif sort:
            output = _sort_gender_token_occurrences(output,
                                                    limit=limit,
                                                    remove_swords=remove_swords)

        self._by_gender[hashed_arguments] = output
        return self._by_gender[hashed_arguments]

    def by_metadata(self, metadata_key, sort=False, diff=False, limit=10, remove_swords=False):
        """
        Merges all adjectives across texts into dictionaries sorted by gender.

        :param metadata_key: a string.
        :param sort: return results in a sorted list.
        :param diff: return the differences between genders.
        :param limit: if sort=True, restrict output to top occurrences sorted.
        :param remove_swords: if sorted=True, remove stop words from results.
        :return: a dictionary of the shape {Gender.label: {str: int , ...}, ...}.

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analysis = GenderTokenAnalysis(corpus)
        >>> list(analysis.by_metadata('author_gender').keys()) == ['male', 'female']
        True
        >>> analysis.by_metadata('author_gender').get('female').get('Female').get('sad')
        7
        >>> analysis.by_metadata('author_gender', sort=True).get('female').get('Female')[0]
        ('sad', 7)
        """

        hashed_arguments = f"{str(metadata_key)}{str(sort)}{str(diff)}{str(limit)}{remove_swords}"

        if self._by_metadata is None:
            self._by_metadata = {}
            self._by_metadata[hashed_arguments] = None
        elif hashed_arguments in self._by_metadata:
            return self._by_metadata[hashed_arguments]

        data = {}

        for document in self.documents:
            metadata_attribute = getattr(document, metadata_key, None)
            if metadata_attribute is None:
                continue

            if metadata_attribute not in data:
                data[metadata_attribute] = {}

            for gender_label in self.gender_labels:
                if gender_label not in data[metadata_attribute]:
                    data[metadata_attribute][gender_label] = []
                data[metadata_attribute][gender_label].append(self[document][gender_label])

        for key in data:
            for gender_label in self.gender_labels:
                data[key][gender_label] = _merge_token_occurrences(data[key][gender_label])
            if diff:
                data[key] = _diff_gender_token_occurrences(data[key],
                                                           limit=limit,
                                                           sort=sort,
                                                           remove_swords=remove_swords)
            elif sort:
                data[key] = _sort_gender_token_occurrences(data[key],
                                                           limit=limit,
                                                           remove_swords=remove_swords)

        self._by_metadata[hashed_arguments] = data
        return self._by_metadata[hashed_arguments]

    def by_overlap(self):
        """
        Looks through the gendered adjectives across the corpus and extracts adjectives that overlap
        across all genders and their occurrences sorted.

        :return: {str: [gender1, gender2, ...], ...}

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> analysis = GenderTokenAnalysis(corpus)
        >>> list(analysis.by_overlap().keys()) == ['sad']
        True
        >>> analysis.by_overlap().get('sad')
        [14, 14]
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

    def by_sorted(self, limit=10, remove_swords=False):
        # pylint: disable=too-many-nested-blocks
        """
        Returns a copy of self with token frequencies sorted.

        :param limit: restrict output to top occurrences sorted.
        :param remove_swords: remove stop words from results.
        :return: a dictionary of the shape {Gender.label: {str: int, ...}, ...}.

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> doc = corpus.documents[-1]
        >>> analysis = GenderTokenAnalysis(corpus)
        >>> list(analysis.by_sorted().keys()) == corpus.documents
        True
        >>> analysis.by_sorted().get(doc).get('Female')
        [('beautiful', 3), ('sad', 1)]
        >>> analysis.by_sorted(limit=1).get(doc).get('Female')
        [('beautiful', 3)]
        """

        hashed_arguments = f"{str(limit)}{remove_swords}"

        if self._by_sorted is None:
            self._by_sorted = {}
            self._by_sorted[hashed_arguments] = None
        elif hashed_arguments in self._by_sorted:
            return self._by_sorted[hashed_arguments]

        output = {}
        for document in self:
            token_frequencies = self[document]
            output[document] = {}
            output[document] = _sort_gender_token_occurrences(token_frequencies,
                                                              limit=limit,
                                                              remove_swords=remove_swords)

        self._by_sorted[hashed_arguments] = output
        return self._by_sorted[hashed_arguments]

    def get_document(self, document, sort=False, diff=False, limit=10, remove_swords=False):
        """
        Returns a dict of the shape {Gender.label: {str: int, ...}, ...}.

        :param document: an instance of the Document class.
        :param sort: boolean, return results in a sorted list.
        :param diff: boolean, return the differences between tokens between Genders.
        :param limit: integer, number of occurrences to return if sorted.
        :param remove_swords: boolean, remove stop words.
        :return: a dictionary of the shape {Gender.label: {str: int, ...}, ...}.

        >>> from corpus_analysis.corpus import Corpus
        >>> from corpus_analysis.testing.common import DOCUMENT_TEST_PATH, DOCUMENT_TEST_CSV
        >>> corpus = Corpus(DOCUMENT_TEST_PATH, csv_path=DOCUMENT_TEST_CSV)
        >>> doc = corpus.documents[-1]
        >>> analysis = GenderTokenAnalysis(corpus)
        >>> analysis.get_document(doc)
        {'Female': {'beautiful': 3, 'sad': 1}, 'Male': {}}
        """

        if document in self:
            output = self.get(document)
            if diff:
                output = _diff_gender_token_occurrences(output,
                                                        limit=limit,
                                                        sort=sort,
                                                        remove_swords=remove_swords)
            elif sort:
                output = _sort_gender_token_occurrences(output,
                                                        limit=limit,
                                                        remove_swords=remove_swords)
            return output
        else:
            raise ValueError('Document not found')

    def store(self, pickle_filepath='gender_tokens_analysis.pgz'):
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
