from more_itertools import windowed
import nltk

from gender_analysis import common
from corpus_analysis.common import store_pickle, load_pickle


def find_gender_adj(document, gender_to_find, word_window=5, genders_to_exclude=None):
    # pylint: disable=too-many-locals
    """
    Takes in a document and a Gender to look for, and returns a dictionary of adjectives that
    appear within a window of 5 words around each identifier

    :param document: Document
    :param gender_to_find: Gender
    :param word_window: number of words to search for in either direction of a gender instance
    :param genders_to_exclude: list of Genders to exclude, or None
    :return: dict of adjectives that appear around pronouns mapped to the number of occurrences

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': \
    '1966', 'filename': 'test_text_7.txt', 'filepath': Path(common.TEST_DATA_PATH, \
    'document_test_files', 'test_text_7.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> find_gender_adj(scarlett, common.MALE, genders_to_exclude=[common.FEMALE])
    {'handsome': 3, 'sad': 1}

    """
    output = {}
    identifiers_to_exclude = []
    text = document.get_tokenized_text()
    adj_tags = ["JJ", "JJR", "JJS"]

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
            if tags[tag_index][1] in adj_tags:
                word = words[tag_index]
                if word in output.keys():
                    output[word] += 1
                else:
                    output[word] = 1

    return output


def find_male_adj(document):
    """
    Takes in a document, returns a dictionary of adjectives that appear within a window of 5
    words around each male pronoun.

    :param: document
    :return: dictionary of adjectives that appear around male pronouns and the number of occurrences

   >>> from corpus_analysis import document
   >>> from pathlib import Path
   >>> from gender_analysis import common
   >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': \
   '1966', 'filename': 'test_text_8.txt', 'filepath': Path(common.TEST_DATA_PATH, \
   'document_test_files', 'test_text_8.txt')}
   >>> scarlett = document.Document(document_metadata)
   >>> find_male_adj(scarlett)
   {'handsome': 3, 'sad': 1}

    """

    return find_gender_adj(document, common.MALE, genders_to_exclude=[common.FEMALE])


def find_female_adj(document):
    """
    Takes in a document, returns a dictionary of adjectives that appear within a window of 5
    words around each female pronoun.

    :param document: A Document object
    :return: dict of adjectives that appear around female pronouns and the number of occurrences

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': \
    '1966', 'filename': 'test_text_9.txt', 'filepath': Path(common.TEST_DATA_PATH, \
    'document_test_files', 'test_text_9.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> find_female_adj(scarlett)
    {'beautiful': 3, 'sad': 1}
    """

    return find_gender_adj(document, common.FEMALE, genders_to_exclude=[common.MALE])


def run_adj_analysis(corpus, gender_list=None):
    """
    Takes in a corpus of novels.
    Return a dictionary with each novel mapped to n dictionaries,
    where n is the number of Genders in gender_list.

    The dictionary contains adjectives:occurrences for each gendered set of identifiers.

    :param corpus: Corpus
    :param gender_list: a list of genders to run the adjective search for.
    :return: dictionary where each key is a novel and the value is len(gender_list)
    dictionaries: Adjectives and number of occurrences associated with gender pronouns

    >>> from corpus_analysis.corpus import Corpus
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.analysis.gender_adjective import run_adj_analysis
    >>> tiny_corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> tiny_results = run_adj_analysis(tiny_corpus)
    >>> tiny_results[[*tiny_results][0]]['Female']['invisible']
    3
    """
    if gender_list is None:
        gender_list = common.BINARY_GROUP

    results = {}

    for document in corpus:
        results[document] = run_adj_analysis_doc(document, gender_list)

    return results


def run_adj_analysis_doc(document, gender_list=None):
    """
    Takes in a document and a list of genders to analyze,
    returns a dictionary with the find_gender_adj results for each gender in gender_list.

    :param document: a Document object
    :param gender_list: a list of genders to run the adjective search for.
    :return: a dictionary of find_gender_adj results for each gender in gender_list.

    >>> from corpus_analysis.corpus import Corpus
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.analysis.gender_adjective import run_adj_analysis_doc
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings = True)
    >>> flatland = corpus.get_document("title", "Flatland")
    >>> adj_dict = run_adj_analysis_doc(flatland)
    >>> adj_dict['Female']['invisible']
    3

    """

    results = {}

    if gender_list is None:
        gender_list = common.BINARY_GROUP

    for gender in gender_list:
        if gender.label == "Female":
            novel_result = find_female_adj(document)
        elif gender.label == "Male":
            novel_result = find_male_adj(document)
        else:
            # Note that we exclude male from female and female from male but do not do this
            # with other genders.
            novel_result = find_gender_adj(document, gender)
        if novel_result != "lower window bound less than 5":
            results.update({gender.label: novel_result})

    return results


def store_raw_results(results, pickle_filepath='pronoun_adj_raw_analysis.pgz'):
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
            store_pickle(results, pickle_filepath)
        else:
            pass
    except IOError:
        store_pickle(results, pickle_filepath)


def merge(novel_adj_dict, full_adj_dict):
    """
    Merges adjective occurrence results from a single novel with results for a larger
    collection of novels.

    :param novel_adj_dict: dictionary of adjectives/#occurrences for one novel
    :param full_adj_dict: dictionary of adjectives/#occurrences for multiple novels
    :return: full_results dictionary with novel_results merged in

    >>> test_novel_adj_dict = {'hello': 5, 'hi': 7, 'hola': 9, 'bonjour': 2}
    >>> test_full_adj_dict = {'hello': 15, 'bienvenue': 3, 'hi': 23}
    >>> merge(test_novel_adj_dict, test_full_adj_dict)
    {'hello': 20, 'bienvenue': 3, 'hi': 30, 'hola': 9, 'bonjour': 2}

    """
    for adj in list(novel_adj_dict.keys()):
        adj_count = novel_adj_dict[adj]
        if adj in list(full_adj_dict.keys()):
            full_adj_dict[adj] += adj_count
        else:
            full_adj_dict[adj] = adj_count
    return full_adj_dict


def merge_raw_results(full_results):
    """
    Merges all adjectives across novels into dictionaries sorted by gender.

    :param full_results: full corpus results from run_adj_analysis
    :return: dictionary in the form {'gender':{'adj': occurrences}}
    >>> from corpus_analysis.corpus import Corpus
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.analysis.gender_adjective import run_adj_analysis
    >>> tiny_corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> tiny_results = run_adj_analysis(tiny_corpus)
    >>> tiny_merged = merge_raw_results(tiny_results)
    >>> tiny_merged['Male']['strong']
    4
    """

    # First, we need to get the genders used in full_results. There's probably a better way to do
    # this, but...

    result_key_list = list(full_results.keys())
    first_key = result_key_list[0]
    genders = list(full_results[first_key].keys())

    merged_results = {}

    for gender in genders:
        merged_results[gender] = {}

    for novel in result_key_list:
        for gender in genders:
            merged_results[gender] = merge(full_results[novel][gender], merged_results[gender])

    return merged_results


def get_overlapping_adjectives_raw_results(merged_results):
    """
    Looks through the gendered adjectives across the corpus and extracts adjectives that overlap
    that overlap across all genders and their occurrences.
    FORMAT - {'adjective': [gender1, gender2, ...]}
    :param merged_results: the results of merge_raw_results.
    :return: dict in form {'adjective': ['gender1': occurrences, 'gender2': occurrences, ... }}

    TODO: Consider adding a more granular function that gets all sub-intersections.
    """

    overlap_results = {}
    genders = list(merged_results.keys())
    sets_of_adjectives = {}

    for gender in genders:
        sets_of_adjectives[gender] = set(list(merged_results[gender].keys()))

    intersects_with_all = set.intersection(*sets_of_adjectives.values())

    for adj in intersects_with_all:
        output = []
        for gender in genders:
            output.append(merged_results[gender][adj])
        overlap_results[adj] = output

    return overlap_results


def results_by_author_gender(full_results):
    """
    Takes in a dictionary of results, and returns nested dictionary that maps author_gender to
    adjective uses by gender.

    :param full_results: dictionary from result of run_adj_analysis
    :return: dictionary in form {'author_gender': {'gender1': {adj:occurrences},\
    'gender2':{adj:occurrences}, ...}}

    """
    data = {}

    result_key_list = list(full_results.keys())
    first_key = result_key_list[0]
    genders = list(full_results[first_key].keys())
    gendered_output = {}

    for gender in genders:
        gendered_output[gender] = {}

    for k in full_results.keys():
        author_gender = getattr(k, 'author_gender', None)
        if author_gender is None:
            continue

        if author_gender not in data:
            data[author_gender] = {}

        for gender in genders:
            data[author_gender][gender] = {}
            data[author_gender][gender] = merge(full_results[k][gender],
                                                data[author_gender][gender])

    return data


def results_by_location(full_results):
    """
    Takes in a dictionary of results, and returns nested dictionary that maps locations to adjective
    uses by gender.

    :param full_results: dictionary from result of run_adj_analysis
    :return: dictionary in form {'location': {'gender1': {adj:occurrences},\
    'gender2':{adj:occurrences}, ...}}
    """

    data = {}

    result_key_list = list(full_results.keys())
    first_key = result_key_list[0]
    genders = list(full_results[first_key].keys())
    gendered_output = {}

    for gender in genders:
        gendered_output[gender] = {}

    for k in full_results.keys():
        location = getattr(k, 'country_publication', None)
        if location is None:
            continue

        if location not in data:
            data[location] = {}

        for gender in genders:
            data[location][gender] = {}
            data[location][gender] = merge(full_results[k][gender], data[location][gender])

    return data


def results_by_date(full_results, time_frame, bin_size):
    """
    Takes in a dictionary of results, returns a dictionary that maps time periods to a
    dictionary mapping adjectives to number of occurrences across novels written in that time period

    :param full_results: dictionary from result of run_adj_analysis
    :param time_frame: tuple (int start year, int end year) for the range of dates to return
    frequencies
    :param bin_size: int for the number of years represented in each list of frequencies
    :return: dictionary in form {date: {gender1: {adj:occurrences}, 'gender2': {adj:occurrences},
     ...}}, where date is the first year in its bin
    """

    data = {}
    result_key_list = list(full_results.keys())
    first_key = result_key_list[0]
    genders = list(full_results[first_key].keys())

    for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
        output = {}
        for gender in genders:
            output[gender] = {}
        data[bin_start_year] = output

    for k in full_results.keys():
        date = getattr(k, 'date', None)
        if date is None:
            continue
        bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
        data[bin_year] = {}
        for gender in genders:
            data[bin_year][gender] = {}
            data[bin_year][gender] = merge(full_results[k][gender], data[bin_year][gender])

    return data


def get_top_adj(full_results, num, remove_swords=False):
    # pylint: disable=too-many-nested-blocks
    """
    Takes dictionary of results from run_adj_analysis and number of top results to return.
    Returns the top num adjectives associated with each gender.

    :param full_results: dictionary from result of run_adj_analysis
    :param num: number of top results to return per gender
    :param remove_swords: Boolean that asks whether to remove English stopwords results
    :return: dictionary of lists of top adjectives associated more with each gender than the others.
    """

    merged_results = merge_raw_results(full_results)
    genders = list(merged_results.keys())

    top = {}

    excluded_results = {}
    for gender in genders:
        excluded_results[gender] = {}
        other_genders = genders.copy()
        other_genders.remove(gender)

        for adj, count in merged_results[gender].items():
            other_count = 0

            for other_gender in other_genders:
                if adj in merged_results[other_gender].keys():
                    if remove_swords:
                        if adj in common.SWORDS_ENG:
                            other_count += merged_results[other_gender][adj]
                    else:
                        other_count += merged_results[other_gender][adj]

            new_count = count - other_count
            excluded_results[gender][adj] = new_count

        # Sorts (adj, count) lists by count.
        top[gender] = sorted(excluded_results[gender].items(),
                             reverse=True, key=lambda x: x[1])[0:num]

    return top


def display_binned_results(metadata_binned_results, num_to_return, remove_swords=False):
    """
    Takes in the results of results_by_(metadata_field) and returns a reader-friendly set of dicts.

    :param metadata_binned_results: The results of results_by_author_gender, location, or date.
    :param num_to_return: Number of adjectives to return
    :param remove_swords: Whether or not to remove English stopwords
    :return: A formatted, sorted dictionary of results.
    """

    display_dict = {}

    for metadata_bin, gender_results in metadata_binned_results.items():
        display_dict[metadata_bin] = {}
        for gender, results in gender_results.items():
            display_dict[metadata_bin][gender] = display_gender_adjectives(results, num_to_return,
                                                                           remove_swords)

    return display_dict


def difference_adjs(gender_adj_dict, num_to_return=10):
    """
    Given result dictionaries from find_gender_adjective,
    returns a dictionary with num_to_return adjectives
    most strongly associated with each gender.

    This works especially well with merged_results from a merge_raw_results function,
    but can also be used on individual locations from result_by_location
    or individual date ranges from result_by_date_range.

    :param gender_adj_dict: a dict of dicts in the form {word:count},
    :param num_to_return: the number of top words to return
    :return: a dict of dicts in the form of {"gender":{list of top words w/ differential counts.

    >>> from corpus_analysis.corpus import Corpus
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> tiny_corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> flatland = tiny_corpus.get_document("title", "Flatland")
    >>> flatland_adjs = run_adj_analysis_doc(flatland)
    >>> flatland_diffs = difference_adjs(flatland_adjs)
    >>> flatland_diffs['Female'][0]
    ('invisible', 3)
    """

    difference_dict = {}

    for gender in gender_adj_dict:
        temp_dict = gender_adj_dict.copy()
        current_gender_dict = temp_dict.pop(gender)
        current_difference = {}

        for word, count in current_gender_dict.items():
            current_difference[word] = count

        for other_gender in temp_dict.keys():
            other_gender_dict = temp_dict[other_gender]

            for word, count in other_gender_dict.items():
                if word in current_difference.keys():
                    current_difference[word] -= count

        stopwordless_words = [
            (key, current_difference[key])
            for key in current_difference
            if key not in common.SWORDS_ENG
        ]

        current_sorted_tuples = sorted(
            stopwordless_words, key=lambda sort_word: sort_word[1], reverse=True
        )

        difference_dict[gender] = current_sorted_tuples[:num_to_return]

    return difference_dict


def display_gender_adjectives(result_dict, num_to_return=10, remove_stopwords=True):
    """
    takes the results of find_gender_adj and prints out to the user the top number_to_return
    adjectives associated with the gender searched for, sorted

    :param result_dict:a dict of word-value frequency
    :param num_to_return: top num of words to be returned to the user -> threshold, or the top?
    :param remove_stopwords: removes English stopwords
    :return: tuples of sorted top num_to_return words with their freq
    """

    if remove_stopwords:
        result_tup_list = [(key, result_dict[key]) for key in result_dict if key not in
                           common.SWORDS_ENG]
    else:
        result_tup_list = [(key, result_dict[key]) for key in result_dict]

    sorted_tuples = sorted(result_tup_list, key=lambda word: word[1], reverse=True)

    return sorted_tuples[:num_to_return]
