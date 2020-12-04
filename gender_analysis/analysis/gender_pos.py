from more_itertools import windowed
import nltk

from gender_analysis import common
pos_dict = {'adj': common.ADJ_TAGS, 'adv': common.ADV_TAGS,
            'proper_noun': common.PROPER_NOUN_TAGS, "verb": common.VERB_TAGS}


def find_gender_pos(document, pos_to_find, gender_to_find, word_window=5, genders_to_exclude=None):
    # pylint: disable=too-many-locals
    """
    Takes in a document, a valid part-of-speech, and a Gender to look for,
    and returns a dictionary of words that appear within a window of 5 words around each identifier.

    :param document: Document
    :param pos_to_find: A valid part of speech tag from pos_dict: ['adj','adv','proper_noun','verb']
    :param word_window: number of words to search for in either direction of a gender instance
    :param gender_to_find: Gender
    :param genders_to_exclude: list of Genders to exclude, or None
    :return: dict of words that appear around pronouns mapped to the number of occurrences

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': \
    '1966', 'filename': 'test_text_7.txt', 'filepath': Path(common.TEST_DATA_PATH, \
    'document_test_files', 'test_text_7.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> find_gender_pos(scarlett, 'adj', common.MALE, genders_to_exclude=[common.FEMALE])
    {'handsome': 3, 'sad': 1}

    """
    output = {}
    identifiers_to_exclude = []
    text = document.get_tokenized_text()

    if pos_to_find in pos_dict.keys():
        pos_tags = pos_dict[pos_to_find]
    else:
        return "Invalid part of speech"

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
            if tags[tag_index][1] in pos_tags:
                word = words[tag_index]
                if word in output.keys():
                    output[word] += 1
                else:
                    output[word] = 1

    return output


def find_male_pos(document, pos_to_find):
    """
    Takes in a document and a pos tag, returns a dictionary of words of the part of speech
    that appear within a window of 5 words around each male pronoun.

    :param document: Document to search for
    :param pos_to_find: A valid part of speech tag from pos_dict: ['adj','adv','proper_noun','verb']
    :return: dictionary of adjectives that appear around male pronouns and the number of occurrences

   >>> from gender_analysis import document
   >>> from pathlib import Path
   >>> from gender_analysis import common
   >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': \
   '1966', 'filename': 'test_text_8.txt', 'filepath': Path(common.TEST_DATA_PATH, \
   'document_test_files', 'test_text_8.txt')}
   >>> scarlett = document.Document(document_metadata)
   >>> find_male_pos(scarlett, 'adj')
   {'handsome': 3, 'sad': 1}
    """

    return find_gender_pos(document, pos_to_find, common.MALE, genders_to_exclude=[common.FEMALE])


def find_female_pos(document, pos_to_find):
    """
    Takes in a document and a pos tag, returns a dictionary of words of the part of speech
    that appear within a window of 5 words around each female pronoun.

    :param document: Document to search through
    :param pos_to_find: A valid part of speech tag from pos_dict: ['adj','adv','proper_noun','verb']
    :return: dictionary of adjectives that appear around male pronouns and the number of occurrences

   >>> from gender_analysis import document
   >>> from pathlib import Path
   >>> from gender_analysis import common
   >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': \
   '1966', 'filename': 'test_text_8.txt', 'filepath': Path(common.TEST_DATA_PATH, \
   'document_test_files', 'test_text_8.txt')}
   >>> scarlett = document.Document(document_metadata)
   >>> find_male_pos(scarlett, 'adj')
   {'handsome': 3, 'sad': 1}

    """

    return find_gender_pos(document, pos_to_find, common.FEMALE, genders_to_exclude=[common.MALE])


def run_pos_analysis(corpus, pos_to_find, gender_list=common.BINARY_GROUP):
    """
    Takes in a corpus of novels.
    Return a dictionary with each novel mapped to n dictionaries,
    where n is the number of Genders in gender_list.

    The dictionary contains adjectives:occurrences for each gendered set of identifiers.

    :param corpus: Corpus
    :param pos_to_find: A valid part of speech tag from pos_dict: ['adj','adv','proper_noun','verb']
    :param gender_list: a list of genders to run the adjective search for.
    :return: dictionary where each key is a novel and the value is len(gender_list)
    dictionaries: Adjectives and number of occurrences associated with gender pronouns

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.analysis.gender_adjective import run_adj_analysis
    >>> tiny_corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> tiny_results = run_pos_analysis(tiny_corpus, 'adj')
    >>> tiny_results[[*tiny_results][0]]['Female']['invisible']
    3
    """

    results = {}

    for document in corpus:
        results[document] = run_pos_analysis_doc(document, pos_to_find, gender_list)

    return results


def run_pos_analysis_doc(document, pos_to_find, gender_list=common.BINARY_GROUP):
    """
    Takes in a document, a pos tag, and a list of genders to analyze,
    returns a dictionary with the find_gender_pos results for each gender in gender_list.

    :param document: a Document object
    :param pos_to_find: A valid part of speech tag from pos_dict: ['adj','adv','proper_noun','verb']
    :param gender_list: a list of genders to run the adjective search for.
    :return: a dictionary of find_gender_adj results for each gender in gender_list.

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.analysis.gender_adjective import run_adj_analysis_doc
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings = True)
    >>> flatland = corpus.get_document("title", "Flatland")
    >>> adj_dict = run_pos_analysis_doc(flatland, 'adj')
    >>> adj_dict['Female']['invisible']
    3

    """

    results = {}

    for gender in gender_list:
        if gender.label == "Female":
            novel_result = find_female_pos(document, pos_to_find)
        elif gender.label == "Male":
            novel_result = find_male_pos(document, pos_to_find)
        else:
            # Note that we exclude male from female and female from male but do not do this
            # with other genders.
            novel_result = find_gender_pos(document, pos_to_find, gender)
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
        common.load_pickle(pickle_filepath)
        user_inp = input("results already stored. overwrite previous analysis? (y/n)")
        if user_inp == 'y':
            common.store_pickle(results, pickle_filepath)
        else:
            pass
    except IOError:
        common.store_pickle(results, pickle_filepath)


def merge(novel_pos_dict, full_pos_dict):
    """
    Merges pos occurrence results from a single novel with results for a larger
    collection of novels.

    :param novel_pos_dict: dictionary of poss/#occurrences for one novel
    :param full_pos_dict: dictionary of poss/#occurrences for multiple novels
    :return: full_results dictionary with novel_results merged in

    >>> test_novel_pos_dict = {'hello': 5, 'hi': 7, 'hola': 9, 'bonjour': 2}
    >>> test_full_pos_dict = {'hello': 15, 'bienvenue': 3, 'hi': 23}
    >>> merge(test_novel_pos_dict, test_full_pos_dict)
    {'hello': 20, 'bienvenue': 3, 'hi': 30, 'hola': 9, 'bonjour': 2}

    """
    for pos in list(novel_pos_dict.keys()):
        pos_count = novel_pos_dict[pos]
        if pos in list(full_pos_dict.keys()):
            full_pos_dict[pos] += pos_count
        else:
            full_pos_dict[pos] = pos_count
    return full_pos_dict


def merge_raw_results(full_results):
    """
    Merges all poss across novels into dictionaries sorted by gender.

    :param full_results: full corpus results from run_pos_analysis
    :return: dictionary in the form {'gender':{'pos': occurrences}}

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


def get_overlapping_pos_raw_results(merged_results):
    """
    Looks through the gendered poss across the corpus and extracts poss that overlap
    that overlap across all genders and their occurrences.
    FORMAT - {'pos': [gender1, gender2, ...]}
    :param merged_results: the results of merge_raw_results.
    :return: dictionary in form {'pos': ['gender1': occurrences, 'gender2': occurences, ... }}

    TODO: Consider adding a more granular function that gets all sub-intersections.

    """
    overlap_results = {}
    genders = list(merged_results.keys())
    sets_of_pos = {}

    for gender in genders:
        sets_of_pos[gender] = set(list(merged_results[gender].keys()))

    intersects_with_all = set.intersection(*sets_of_pos.values())

    for pos in intersects_with_all:
        output = []
        for gender in genders:
            output.append(merged_results[gender][pos])
        overlap_results[pos] = output

    return overlap_results


def results_by_author_gender(full_results):
    """
    Takes in a dictionary of results, and returns nested dictionary that maps author_gender to
    part_of_speech uses by gender.

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


def get_top_pos(full_results, num, remove_swords=False):
    # pylint: disable=too-many-nested-blocks
    """
    Takes dictionary of results from run_pos_analysis and number of top results to return.
    Returns the top num adjectives associated with each gender.

    :param full_results: dictionary from result of run_pos_analysis
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

        for pos_word, count in merged_results[gender].items():
            other_count = 0

            for other_gender in other_genders:
                if pos_word in merged_results[other_gender].keys():
                    if remove_swords:
                        if pos_word in common.SWORDS_ENG:
                            other_count += merged_results[other_gender][pos_word]
                    else:
                        other_count += merged_results[other_gender][pos_word]

            new_count = count - other_count
            excluded_results[gender][pos_word] = new_count

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
            display_dict[metadata_bin][gender] = display_gender_pos(results, num_to_return,
                                                                    remove_swords)

    return display_dict


def difference_pos(gender_pos_dict, num_to_return=10):
    """
    Given result dictionaries from find_gender_pos,
    returns a dictionary with num_to_return pos_words
    most strongly associated with each gender.

    This works especially well with merged_results from a merge_raw_results function,
    but can also be used on individual locations from result_by_location
    or individual date ranges from result_by_date_range.

    :param gender_pos_dict: a dict of dicts in the form {word:count},
    :param num_to_return: the number of top words to return
    :return: a dict of dicts in the form of {"gender":{list of top words w/ differential counts.

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> tiny_corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> flatland = tiny_corpus.get_document("title", "Flatland")
    >>> flatland_pos = run_pos_analysis_doc(flatland, 'adj')
    >>> flatland_diffs = difference_pos(flatland_pos)
    >>> flatland_diffs['Female'][0]
    ('invisible', 3)
    """

    difference_dict = {}

    for gender in gender_pos_dict:
        temp_dict = gender_pos_dict.copy()
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


def display_gender_pos(result_dict, num_to_return=10, remove_stopwords=True):
    """
    takes the results of find_gender_pos and prints out to the user the top number_to_return
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
