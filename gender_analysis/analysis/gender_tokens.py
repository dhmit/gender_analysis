from more_itertools import windowed
import nltk
from collections import Counter, UserDict

from gender_analysis.corpus import Corpus
from gender_analysis.document import Document
from gender_analysis.common import load_pickle, store_pickle, MALE, FEMALE, BINARY_GROUP, SWORDS_ENG
from gender_analysis.gender import Gender

class GenderTokenAnalysis(UserDict):
    """
    The GenderTokenAnalysis instance is a dictionary of the shape:
    { Document: { str(Gender.label): { str(token): int } } }
    """

    def __init__(self, dictionary, documents, genders, tokens):
        self.documents = documents
        self.genders = genders
        self.gender_labels = [gender.label for gender in genders]
        self.tokens = tokens
        self._by_date = None
        self._by_differences = None
        self._by_gender = None
        self._by_metadata = None
        self._by_overlap = None
        self._by_sorted = None
        super().__init__(dictionary)

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
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_date((2000, 2040), 10).keys()) == [2000, 2010, 2020, 2030]
        True
        >>> analysis.by_date((2000, 2040), 10)[2010]['Female']['jet']
        1
        >>> analysis.by_date((2000, 2040), 10, sort=True)[2010]['Female'][0]
        ('jet', 1)
        """
        hashed_dates = f'{str(time_frame[0])}{str(time_frame[1])}{str(bin_size)}'
        hashed_options = f'{str(sort)}{str(diff)}{str(limit)}{remove_swords}'
        hashed_arguments = hashed_dates + hashed_options

        if self._by_date is None:
            self._by_date = {}
            self._by_date[hashed_arguments] = None
        elif hashed_arguments in self._by_date:
            return self._by_date[hashed_arguments]

        data = {}

        for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
            output = {}
            for gender_label in self.gender_labels:
                output[gender_label] = {}
            data[bin_start_year] = output

        for k in self.documents:
            date = getattr(k, 'date', None)
            if date is None:
                continue
            bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
            if bin_year not in data:
                data[bin_year] = {}
            merged_token_frequencies = {}
            for gender_label in self.gender_labels:
                if gender_label not in data[bin_year]:
                    data[bin_year][gender_label] = {}
                merged_token_frequencies[gender_label] = {}
                merged_token_frequencies[gender_label] = _merge_token_frequencies(
                    [self[k][gender_label], data[bin_year][gender_label]]
                )

            data[bin_year] = merged_token_frequencies

        if sort:
            for date, gender_token_frequencies in data.items():
                data[date] = _sort_gender_token_frequencies(gender_token_frequencies,
                                                            limit=limit,
                                                            remove_swords=remove_swords)
        elif diff:
            for date, gender_token_frequencies in data.items():
                data[date] = _diff_gender_token_frequencies(gender_token_frequencies,
                                                            limit=limit,
                                                            remove_swords=remove_swords)

        self._by_date[hashed_arguments] = data
        return self._by_date[hashed_arguments]

    def by_differences(self, limit=10, remove_swords=False):
        """
        TODO: write out a more accurate description for this.

        :param limit: if sort=True, restrict output to top occurrences sorted.
        :param remove_swords: if sort=True, remove stop words from results.
        :return: a dictionary of the shape { str(Gender.label): [ ( str(token), int ) ] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> analysis.by_differences().keys() == analysis.keys()
        True
        >>> analysis.by_differences().get(corpus.documents[0]).get('Male')[0]
        ('stay', 1)
        """

        hashed_arguments = f"{str(limit)}{remove_swords}"

        if self._by_differences is None:
            self._by_differences = {}
            self._by_differences[hashed_arguments] = None
        elif hashed_arguments in self._by_differences:
            return self._by_differences[hashed_arguments]

        diff = {}

        for document in self:
            diff[document] = {}
            diff[document] = _diff_gender_token_frequencies(self[document],
                                                            limit=limit,
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
        :return: a dictionary of the shape { str(Gender.label): { str(token): int } } or
                 { str(Gender.label): [(str(token), int)] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_gender().keys()) == ['Female', 'Male']
        True
        >>> analysis.by_gender().get('Female').get('time')
        2
        >>> analysis.by_gender(sort=True).get('Male')[0]
        ('road', 2)
        """

        hashed_arguments = f"{str(sort)}{str(diff)}{str(limit)}{remove_swords}"

        if self._by_gender is None:
            self._by_gender = {}
            self._by_gender[hashed_arguments] = None
        elif hashed_arguments in self._by_gender:
            return self._by_gender[hashed_arguments]

        merged_results = {}
        for gender_label in self.gender_labels:
            current_gender_token_frequencies = [self[document][gender_label] for document in self]
            merged_results[gender_label] = {}
            merged_results[gender_label] = _merge_token_frequencies(current_gender_token_frequencies)

        output = merged_results

        if sort:
            output = _sort_gender_token_frequencies(output,
                                                    limit=limit,
                                                    remove_swords=remove_swords)
        elif diff:
            output = _diff_gender_token_frequencies(output,
                                                    limit=limit,
                                                    remove_swords=remove_swords)

        self._by_gender[hashed_arguments] = output
        return self._by_gender[hashed_arguments]

    def by_metadata(self, metadata_key, sort=False, diff=False, limit=10, remove_swords=False):
        """
        Merges all adjectives across texts into dictionaries sorted by gender.

        :param metadata_key: a string
        :param sort: return results in a sorted list.
        :param diff: return the differences between genders.
        :param limit: if sort=True, restrict output to top occurrences sorted.
        :param remove_swords: if sorted=True, remove stop words from results.
        :return: a dictionary of the shape { Gender: { str: int } } or { Gender: [(str, int)] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_metadata('author_gender').keys()) == ['male', 'female']
        True
        >>> analysis.by_metadata('author_gender').get('female').get('Female').get('time')
        2
        >>> analysis.by_metadata('author_gender', sort=True).get('female').get('Female')[0]
        ('time', 2)
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
                data[key][gender_label] = _merge_token_frequencies(data[key][gender_label])
            if sort:
                data[key] = _sort_gender_token_frequencies(data[key],
                                                           limit=limit,
                                                           remove_swords=remove_swords)
            elif diff:
                data[key] = _diff_gender_token_frequencies(data[key],
                                                           limit=limit,
                                                           remove_swords=remove_swords)

        self._by_metadata[hashed_arguments] = data
        return self._by_metadata[hashed_arguments]

    def by_overlap(self):
        """
        Looks through the gendered adjectives across the corpus and extracts adjectives that overlap
        across all genders and their occurrences sorted.

        :return: { str: [gender1, gender2, ...] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_overlap().keys()) == ['i']
        True
        >>> analysis.by_overlap().get('i')
        [1, 1]
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
        Returns self with token frequencies replaced by a list of occurrences sorted.

        :param limit: restrict output to top occurrences sorted.
        :param remove_swords: remove stop words from results.
        :return: a dictionary of the shape { Gender: { str: int } } or { Gender: [(str, int)] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_sorted().keys()) == corpus.documents
        True
        >>> analysis.by_sorted()[corpus.documents[0]].get('Male')[0]
        ('stay', 1)
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
            output[document] = _sort_gender_token_frequencies(token_frequencies,
                                                              limit=limit,
                                                              remove_swords=remove_swords)

        self._by_sorted[hashed_arguments] = output
        return self._by_sorted[hashed_arguments]

    def get_document(self, name, sort=False, limit=10, remove_swords=False):
        """
        Returns a dict of the shape { str: { str: int } }.

        :param name: the name (filename minus extension) of the Document.
        :param sort: return results in a sorted list.
        :param limit: restrict output to top occurrences sorted.
        :param remove_swords: remove stop words from results.
        :return: a dictionary of the shape { Gender: { str: int } } or { Gender: [(str, int)] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> analysis.get_document('_control1_text1')
        {'Female': {}, 'Male': {'stay': 1, 'road': 1, 'work': 1, 'abstraction': 1}}
        """
        for document in self.documents:
            if document.filename[0:len(document.filename) - 4] == name:
                if sort:
                    output = _sort_gender_token_frequencies(self[document],
                                                            limit=limit,
                                                            remove_swords=remove_swords)
                else:
                    output = self[document]
                return output

        raise ValueError('name must be a valid Document name')

    def store(self, pickle_filepath='pronoun_adj_raw_analysis.pgz'):
        """
        Saves the results from run_adj_analysis to a pickle file.

        :param results: dictionary of results from run_adj_analysis
        :param pickle_filepath: filepath to save the output
        :return: None, saves results as pickled file with name 'pronoun_adj_raw_analysis'

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


def _generate_token_frequencies(document, gender_to_find, word_window, tokens, genders_to_exclude=None):
    """

    :param document: an instance of the Document class
    :param gender_to_find: an instance of the Gender class
    :param word_window: number of words to search for in either direction of a Gender instance
    :param tokens: a list containing NLTK token strings
    :param genders_to_exclude: a list containing instances of the Gender class
    :return: an instance of a dict of the shape { str: int }

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.common import MALE, FEMALE
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> doc = corpus.documents[0]
    >>> test = _generate_token_frequencies(doc, MALE, 5, ['NN'], genders_to_exclude=[FEMALE])
    >>> test.get('eye')
    6
    """

    # pylint: disable=too-many-locals
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


def _generate_gender_token_frequencies(document, genders, tokens, word_window=5):
    """

    :param document: an instance of the Document class
    :param genders: a list containing instances of the Gender class
    :param tokens: a list containing NLTK token strings
    :param word_window: number of words to search for in either direction of a Gender instance
    :return: an instance of a dict of the shape { GENDER: { str: int } }

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.common import BINARY_GROUP
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> doc = corpus.documents[0]
    >>> test = _generate_gender_token_frequencies(doc, BINARY_GROUP, ['NN'])
    >>> list(test.keys()) == ['Female', 'Male']
    True
    >>> test.get('Female').get('woman')
    4
    """

    results = {}

    for gender in genders:
        if gender.label == FEMALE.label:
            novel_result = _generate_token_frequencies(document,
                                                     FEMALE,
                                                     word_window,
                                                     tokens,
                                                     genders_to_exclude=[MALE])
        elif gender.label == MALE.label:
            novel_result = _generate_token_frequencies(document,
                                                     MALE,
                                                     word_window,
                                                     tokens,
                                                     genders_to_exclude=[FEMALE])
        else:
            # Note that we exclude male from female and female from male but do not do this
            # with other genders.
            novel_result = _generate_token_frequencies(document, gender, word_window, tokens)
        if novel_result != "lower window bound less than 5":
            results.update({gender.label: novel_result})

    return results


def _diff_gender_token_frequencies(gender_token_frequencies, limit=10, remove_swords=False):
    """
    Returns a dictionary with number of occurrences of adjectives
    most strongly associated with each gender.
    >>> token_frequency1 = { 'foo': 1, 'bar': 2 }
    >>> token_frequency2 = { 'foo': 2, 'baz': 3 }
    >>> test = { 'Male': token_frequency1, 'Female': token_frequency2 }
    >>> by_differences = _diff_gender_token_frequencies(test)
    >>> by_differences['Male']
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

    return _sort_gender_token_frequencies(difference_dict,
                                          limit=limit,
                                          remove_swords=remove_swords)


def _merge_token_frequencies(token_frequencies):
    """

    :param token_frequencies: a list containing instances of dicts of the shape { str: int }
    :return: a dict of the same shape as the above, merged.

    >>> test_1 = { 'good': 1, 'bad': 1, 'ugly': 1 }
    >>> test_2 = { 'good': 3, 'bad': 0, 'weird': 2 }
    >>> test_3 = { 'good': 2, 'bad': 4, 'weird': 0, 'ugly': 2 }
    >>> merged_token_frequency = _merge_token_frequencies([test_1, test_2, test_3])
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


def _sort_token_frequencies(token_frequencies, limit=None, remove_swords=False):
    """
    Transforms a dict of type { str: int } into a sorted list of len limit and shape [(str, int)].

    :param limit: number of results to return.
    :param remove_swords: remove stop words from results.
    :return: a list of the shape [(str, int)]
    >>> example = { 'foo': 1, 'bar': 3, 'bat': 4 }
    >>> _sort_token_frequencies(example)
    [('bat', 4), ('bar', 3), ('foo', 1)]
    """
    output = token_frequencies.copy()

    if remove_swords:
        output = {key: count for key, count in output.items() if key not in SWORDS_ENG}
    sorted_counter = Counter(output).most_common()

    if limit is None:
        limit = len(sorted_counter)
    return sorted_counter[:limit]


def _sort_gender_token_frequencies(gender_token_frequencies, limit=10, remove_swords=False):
    """
    Returns a dictionary with number of occurrences of adjectives sorted.
    >>> token_frequency1 = { 'foo': 1, 'bar': 2 }
    >>> token_frequency2 = { 'foo': 2, 'baz': 3 }
    >>> test = { 'Male': token_frequency1, 'Female': token_frequency2 }
    >>> by_occurrences = _sort_gender_token_frequencies(test)
    >>> by_occurrences['Male']
    [('bar', 2), ('foo', 1)]
    >>> by_occurrences['Female']
    [('baz', 3), ('foo', 2)]
    """
    output = {}
    for gender, token_frequencies in gender_token_frequencies.items():
        output[gender] = {}
        output[gender] = _sort_token_frequencies(token_frequencies,
                                                 limit=limit,
                                                 remove_swords=remove_swords)
    return output


def generate_analysis(texts, tokens, genders=BINARY_GROUP):
    """
    Accepts a Corpus, Document, or list of Documents and an optional list of Gender instances
    and creates a dict of the subclass GenderTokenAnalysis.

    :param texts: Corpus instance, Document instance, or list of Document instances.
    :param tokens: list of Gender instances.
    :param genders: list of Gender instances.
    :return: a dict of subclass GenderTokenAnalysis

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> doc = corpus.documents[0]
    >>> analysis1 = generate_analysis(corpus, ['NN'])
    >>> analysis2 = generate_analysis(corpus.documents, ['NN'])
    >>> analysis3 = generate_analysis([document for document in corpus.documents], ['NN'])
    >>> analysis4 = generate_analysis(doc, ['NN'])
    >>> isinstance(analysis1, GenderTokenAnalysis)
    True
    >>> isinstance(analysis2, GenderTokenAnalysis)
    True
    >>> isinstance(analysis3, GenderTokenAnalysis)
    True
    >>> isinstance(analysis4, GenderTokenAnalysis)
    True
    >>> generate_analysis('some garbage', ['NN'])
    Traceback (most recent call last):
    ValueError: texts must be of type Document, Corpus, or a list of Documents
    >>> generate_analysis(doc, 'some garbage')
    Traceback (most recent call last):
    ValueError: tokens must be a list of NLTK token strings
    >>> generate_analysis(doc, [doc])
    Traceback (most recent call last):
    ValueError: all items in list tokens must be of type str
    >>> generate_analysis(doc, ['NN'], genders=['some garbage'])
    Traceback (most recent call last):
    ValueError: all items in list genders must be of type Gender
    """

    analysis_dictionary = {}

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

    for document in sanitized_documents:
        analysis_dictionary[document] = _generate_gender_token_frequencies(document, genders, tokens)

    return GenderTokenAnalysis(analysis_dictionary, sanitized_documents, genders, tokens)


def generate_adjective_analysis(texts, genders=BINARY_GROUP):
    """
    Accepts a Corpus, Document, or list of Documents and an optional list of Gender instances
    and calls generate_analysis with NLTK adjective tokens, returning a GenderTokenAnalaysis.

    :param texts: Corpus instance, Document instance, or list of Document instances.
    :param genders: list of Gender instances.
    :return: a dict of subclass GenderTokenAnalysis

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> analysis = generate_adjective_analysis(corpus)
    >>> isinstance(analysis, GenderTokenAnalysis)
    True
    """
    return generate_analysis(texts, tokens=["JJ", "JJR", "JJS"], genders=genders)


def generate_noun_analysis(texts, genders=BINARY_GROUP):
    """
    Accepts a Corpus, Document, or list of Documents and an optional list of Gender instances
    and calls generate_analysis with NLTK noun tokens, returning a GenderTokenAnalaysis.

    :param texts: Corpus instance, Document instance, or list of Document instances.
    :param genders: list of Gender instances.
    :return: a dict of subclass GenderTokenAnalysis

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> analysis = generate_noun_analysis(corpus)
    >>> isinstance(analysis, GenderTokenAnalysis)
    True

    """
    return generate_analysis(texts, tokens=["NN"], genders=genders)
