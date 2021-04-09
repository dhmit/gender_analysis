from more_itertools import windowed
import nltk
from collections import Counter, UserDict, UserList

from gender_analysis.corpus import Corpus
from gender_analysis.document import Document
from gender_analysis.common import load_pickle, store_pickle, MALE, FEMALE, BINARY_GROUP, SWORDS_ENG
from gender_analysis.gender import Gender


class TokenFrequency(UserDict):
    """
    A dictionary of the shape: { str: int }
    with one predefined helper methods.
    """

    def by_occurrences(self, occurrences=None, remove_swords=False):
        """
        Transforms a dict of type { str: int } into a sorted list of len occurrences.

        :param occurrences: number of results to return.
        :param remove_swords: remove stop words from results.
        :return: a list of the shape [(str, int)]
        >>> test = TokenFrequency({ 'foo': 1, 'bar': 3, 'bat': 4 })
        >>> test.by_occurrences()
        [('bat', 4), ('bar', 3), ('foo', 1)]
        """
        output = self.copy()

        if remove_swords:
            output = {key: count for key, count in output.items() if key not in SWORDS_ENG}
        sorted_counter = Counter(output).most_common()

        if occurrences is None:
            occurrences = len(sorted_counter)
        return sorted_counter[:occurrences]


class GenderTokenOccurrences(UserDict):
    """
    A dictionary of the shape: { Gender: list }
    with a few predefined helper methods.
    """

    def by_female(self) -> list:
        return self[FEMALE]

    def by_male(self) -> list:
        return self[MALE]


class GenderTokenFrequencies(UserDict):
    """
    A dictionary of the shape: { Gender: TokenFrequency }
    with a few predefined helper methods.
    """

    def by_differences(self, occurrences=10, remove_swords=False):
        """
        Returns a dictionary with number of occurrences of adjectives
        most strongly associated with each gender.
        >>> from gender_analysis.common import MALE, FEMALE
        >>> token_frequency1 = TokenFrequency({ 'foo': 1, 'bar': 2 })
        >>> token_frequency2 = TokenFrequency({ 'foo': 2, 'baz': 3 })
        >>> test = GenderTokenFrequencies({ MALE: token_frequency1, FEMALE: token_frequency2 })
        >>> by_differences = test.by_differences()
        >>> by_differences[MALE]
        [('bar', 2), ('foo', -1)]
        """

        if not isinstance(occurrences, int):
            raise ValueError('occurrences must be of type int')

        difference_dict = {}

        for gender in self:
            current_difference = {}

            for word, count in self[gender].items():
                current_difference[word] = count

            for other_gender in self:
                if other_gender == gender:
                    continue
                other_adjective_frequency = self[other_gender]

                for word, count in other_adjective_frequency.items():
                    if word in current_difference.keys():
                        current_difference[word] -= count

            difference_dict[gender] = TokenFrequency(current_difference)

        difference_dict = GenderTokenFrequencies(difference_dict)

        return difference_dict.by_occurrences(occurrences=occurrences, remove_swords=remove_swords)

    def by_occurrences(self, occurrences=10, remove_swords=False):
        """
        Returns a dictionary with number of occurrences of adjectives sorted.
        >>> from gender_analysis.common import MALE, FEMALE
        >>> token_frequency1 = TokenFrequency({ 'foo': 1, 'bar': 2 })
        >>> token_frequency2 = TokenFrequency({ 'foo': 2, 'baz': 3 })
        >>> test = GenderTokenFrequencies({ MALE: token_frequency1, FEMALE: token_frequency2 })
        >>> by_occurrences = test.by_occurrences()
        >>> by_occurrences[MALE]
        [('bar', 2), ('foo', 1)]
        >>> by_occurrences[FEMALE]
        [('baz', 3), ('foo', 2)]
        """
        output = {}
        for gender, token_frequency in self.items():
            output[gender] = {}
            output[gender] = token_frequency.by_occurrences(occurrences=occurrences,
                                                            remove_swords=remove_swords)
        return GenderTokenOccurrences(output)

    def by_female(self) -> TokenFrequency:
        return self[FEMALE]

    def by_male(self) -> TokenFrequency:
        return self[MALE]


class GenderTokenAnalysis(UserDict):
    """
    The GenderTokenAnalysis instance is a dictionary of the shape:
    { Document: GenderTokenFrequencies }
    """

    def __init__(self, dictionary, documents, genders, tokens):
        self.documents = documents
        self.genders = genders
        self.tokens = tokens
        self._by_date = None
        self._by_differences = None
        self._by_gender = None
        self._by_metadata = None
        self._by_occurrences = None
        self._by_overlap = None
        super().__init__(dictionary)

    def by_date(self, time_frame, bin_size, by_occurrences=False, occurrences=10, remove_swords=False):
        """
        Return analysis in the format { date(int): GenderTokenFrequencies }

        :param time_frame: a tuple of the format (start_date, end_date).
        :param bin_size: int for the number of years represented in each list of frequencies
        :param by_occurrences: return results in a sorted list.
        :param occurrences: if sorted=True, restrict output to top occurrences.
        :param remove_swords: if sorted=True, remove stop words from results.
        :return: a dictionary of the shape { Gender: { str: int } } or { Gender: [(str, int)] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_date((2000, 2040), 10).keys()) == [2000, 2010, 2020, 2030]
        True
        >>> analysis.by_date((2000, 2040), 10)[2010][FEMALE]['jet']
        1
        >>> analysis.by_date((2000, 2040), 10, by_occurrences=True)[2010][FEMALE][0]
        ('jet', 1)
        """
        hashed_dates = f'{str(time_frame[0])}{str(time_frame[1])}{str(bin_size)}'
        hashed_options = f'{str(by_occurrences)}{str(occurrences)}{remove_swords}'
        hashed_arguments = hashed_dates + hashed_options

        if self._by_date is None:
            self._by_date = {}
            self._by_date[hashed_arguments] = None
        elif hashed_arguments in self._by_date:
            return self._by_date[hashed_arguments]

        data = {}

        for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
            output = {}
            for gender in self.genders:
                output[gender] = {}
            data[bin_start_year] = GenderTokenFrequencies(output)

        for k in self.documents:
            date = getattr(k, 'date', None)
            if date is None:
                continue
            bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
            data[bin_year] = {}
            merged_token_frequencies = {}
            for gender in self.genders:
                if gender not in data[bin_year]:
                    data[bin_year][gender] = {}
                merged_token_frequencies[gender] = {}
                merged_token_frequencies[gender] = _merge(
                    [self[k][gender], data[bin_year][gender]]
                )

            token_frequencies = GenderTokenFrequencies(merged_token_frequencies)
            if by_occurrences:
                token_frequencies = token_frequencies.by_occurrences(occurrences=occurrences,
                                                                     remove_swords=remove_swords)
            data[bin_year] = token_frequencies

        self._by_date[hashed_arguments] = data
        return self._by_date[hashed_arguments]

    def by_differences(self, occurrences=10, remove_swords=False):
        """
        Merges all adjectives across texts into dictionaries sorted by gender.

        :param occurrences: if sorted=True, restrict output to top occurrences.
        :param remove_swords: if sorted=True, remove stop words from results.
        :return: a dictionary of the shape { Gender: [ ( str, int ) ] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> analysis.by_differences().keys() == analysis.keys()
        True
        >>> analysis.by_differences()[corpus.documents[0]][MALE][0]
        ('stay', 1)
        """

        hashed_arguments = f"{str(occurrences)}{remove_swords}"

        if self._by_differences is None:
            self._by_differences = {}
            self._by_differences[hashed_arguments] = None
        elif hashed_arguments in self._by_differences:
            return self._by_differences[hashed_arguments]

        output = {}

        for document in self:
            output[document] = {}
            output[document] = self[document].by_differences(occurrences=occurrences,
                                                             remove_swords=remove_swords)

        self._by_differences[hashed_arguments] = output
        return self._by_differences[hashed_arguments]

    def by_gender(self, by_occurrences=False, occurrences=10, remove_swords=False):
        """
        Merges all adjectives across texts into dictionaries sorted by gender.

        :param by_occurrences: return results in a sorted list.
        :param occurrences: if sorted=True, restrict output to top occurrences.
        :param remove_swords: if sorted=True, remove stop words from results.
        :return: a dictionary of the shape { Gender: { str: int } } or { Gender: [(str, int)] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_gender().keys()) == [FEMALE, MALE]
        True
        >>> analysis.by_gender()[FEMALE]['time']
        2
        >>> analysis.by_gender(by_occurrences=True)[MALE][0]
        ('road', 2)
        """

        hashed_arguments = f"{str(by_occurrences)}{str(occurrences)}{remove_swords}"

        if self._by_gender is None:
            self._by_gender = {}
            self._by_gender[hashed_arguments] = None
        elif hashed_arguments in self._by_gender:
            return self._by_gender[hashed_arguments]

        merged_results = {}
        for gender in self.genders:
            current_gender_token_frequencies = [self[document][gender] for document in self]
            merged_results[gender] = {}
            merged_results[gender] = _merge(current_gender_token_frequencies)

        output = GenderTokenFrequencies(merged_results)

        if by_occurrences:
            output = output.by_occurrences(occurrences=occurrences, remove_swords=remove_swords)

        self._by_gender[hashed_arguments] = output
        return self._by_gender[hashed_arguments]

    def by_metadata(self, metadata_key, by_occurrences=False, occurrences=10, remove_swords=False):
        """
        Merges all adjectives across texts into dictionaries sorted by gender.

        :param metadata_key: a string
        :param by_occurrences: return results in a sorted list.
        :param occurrences: if sorted=True, restrict output to top occurrences.
        :param remove_swords: if sorted=True, remove stop words from results.
        :return: a dictionary of the shape { Gender: { str: int } } or { Gender: [(str, int)] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_metadata('author_gender').keys()) == ['male', 'female']
        True
        >>> analysis.by_metadata('author_gender')['female'][FEMALE]['time']
        2
        >>> analysis.by_metadata('author_gender', by_occurrences=True)['female'][FEMALE][0]
        ('time', 2)
        """

        hashed_arguments = f"{str(metadata_key)}{str(by_occurrences)}{str(occurrences)}{remove_swords}"

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

            for gender in self.genders:
                if gender not in data[metadata_attribute]:
                    data[metadata_attribute][gender] = []
                data[metadata_attribute][gender].append(self[document][gender])

        for key in data:
            for gender in self.genders:
                data[key][gender] = _merge(data[key][gender])
            token_frequencies = GenderTokenFrequencies(data[key])
            if by_occurrences:
                data[key] = token_frequencies.by_occurrences(occurrences=occurrences,
                                                             remove_swords=remove_swords)
            else:
                data[key] = token_frequencies

        self._by_metadata[hashed_arguments] = data
        return self._by_metadata[hashed_arguments]

    def by_occurrences(self, occurrences=10, remove_swords=False):
        # pylint: disable=too-many-nested-blocks
        """
        Returns self with TokenFrequencies replaced by a list of occurrences.

        :param occurrences: restrict output to top occurrences.
        :param remove_swords: remove stop words from results.
        :return: a dictionary of the shape { Gender: { str: int } } or { Gender: [(str, int)] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_occurrences().keys()) == corpus.documents
        True
        >>> analysis.by_occurrences()[corpus.documents[0]][MALE][0]
        ('stay', 1)
        """

        hashed_arguments = f"{str(occurrences)}{remove_swords}"

        if self._by_occurrences is None:
            self._by_occurrences = {}
            self._by_occurrences[hashed_arguments] = None
        elif hashed_arguments in self._by_occurrences:
            return self._by_occurrences[hashed_arguments]

        output = {}
        for document in self:
            token_frequencies = GenderTokenFrequencies(self[document])
            output[document] = {}
            output[document] = token_frequencies.by_occurrences(occurrences=occurrences,
                                                                remove_swords=remove_swords)

        self._by_occurrences[hashed_arguments] = output
        return self._by_occurrences[hashed_arguments]

    def by_overlap(self):
        """
        Looks through the gendered adjectives across the corpus and extracts adjectives that overlap
        across all genders and their occurrences.

        :return: { str: [gender1, gender2, ...] }
        >>> from gender_analysis.corpus import Corpus
        >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, CONTROLLED_TEST_CORPUS_CSV
        >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=CONTROLLED_TEST_CORPUS_CSV, ignore_warnings=True)
        >>> analysis = generate_analysis(corpus, ['NN'])
        >>> list(analysis.by_overlap().keys()) == ['i']
        True
        >>> analysis.by_overlap()['i']
        [1, 1]
        """

        if self._by_overlap is not None:
            return self._by_overlap

        overlap_results = {}
        sets_of_adjectives = {}

        for gender in self.genders:
            sets_of_adjectives[gender] = set(list(self.by_gender()[gender].keys()))

        intersects_with_all = set.intersection(*sets_of_adjectives.values())

        for adj in intersects_with_all:
            output = []
            for gender in self.genders:
                output.append(self.by_gender()[gender][adj])
            overlap_results[adj] = output

        self._by_overlap = overlap_results
        return self._by_overlap

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


def _merge(token_frequencies):
    """

    :param token_frequencies: a list containing instances of TokenFrequency
    :return: a dict of the subclass TokenFrequency

    >>> token_frequency_1 = TokenFrequency({ 'good': 1, 'bad': 1, 'ugly': 1 })
    >>> token_frequency_2= TokenFrequency({ 'good': 3, 'bad': 0, 'weird': 2 })
    >>> token_frequency_3= TokenFrequency({ 'good': 2, 'bad': 4, 'weird': 0, 'ugly': 2 })
    >>> merged_token_frequency = _merge([token_frequency_1, token_frequency_2, token_frequency_3])
    >>> isinstance(merged_token_frequency, TokenFrequency)
    True
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
    return TokenFrequency(merged_token_frequencies)


def _generate_token_frequency(document, gender_to_find, word_window, tokens, genders_to_exclude=None):
    """

    :param document: an instance of the Document class
    :param gender_to_find: an instance of the Gender class
    :param word_window: number of words to search for in either direction of a Gender instance
    :param tokens: a list containing NLTK token strings
    :param genders_to_exclude: a list containing instances of the Gender class
    :return: an instance of the dict subclass TokenFrequency

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.common import MALE, FEMALE
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> doc = corpus.documents[0]
    >>> token_frequency = _generate_token_frequency(doc, MALE, 5, ['NN'], genders_to_exclude=[FEMALE])
    >>> isinstance(token_frequency, TokenFrequency)
    True
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

    return TokenFrequency(output)


def _generate_token_frequencies(document, genders, tokens, word_window=5):
    """

    :param document: an instance of the Document class
    :param genders: a list containing instances of the Gender class
    :param tokens: a list containing NLTK token strings
    :param word_window: number of words to search for in either direction of a Gender instance
    :return: an instance of the dict subclass GenderTokenFrequencies

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.common import BINARY_GROUP
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> doc = corpus.documents[0]
    >>> gender_token_frequency = _generate_token_frequencies(doc, BINARY_GROUP, ['NN'])
    >>> isinstance(gender_token_frequency, GenderTokenFrequencies)
    True
    """

    results = {}

    for gender in genders:
        if gender.label == FEMALE.label:
            novel_result = _generate_token_frequency(document, FEMALE, word_window, tokens, genders_to_exclude=[MALE])
        elif gender.label == MALE.label:
            novel_result = _generate_token_frequency(document, MALE, word_window, tokens, genders_to_exclude=[FEMALE])
        else:
            # Note that we exclude male from female and female from male but do not do this
            # with other genders.
            novel_result = _generate_token_frequency(document, gender, word_window, tokens)
        if novel_result != "lower window bound less than 5":
            results.update({gender: novel_result})

    return GenderTokenFrequencies(results)


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
        analysis_dictionary[document] = _generate_token_frequencies(document, genders, tokens)
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
