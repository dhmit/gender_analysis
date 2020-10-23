from statistics import median
from more_itertools import windowed
import nltk

from gender_analysis.analysis.instance_distance import words_instance_dist
from gender_analysis import common

pos_dict = {'adj':common.ADJ_TAGS, 'adv':common.ADV_TAGS, 'proper_noun':common.PROPER_NOUN_TAGS, "verb":common.VERB_TAGS}

def find_gender_pos(document, pos_to_find, gender_to_find, genders_to_exclude = []):
    """
    Takes in a document, a valid part-of-speech, and a Gender to look for, and returns a
    dictionary of  that appear within a window of 5 words around each identifier

    :param: document: Document
    :param: gender_to_find: Gender
    :param: genders_to_exclude: list of Genders to exclude.
    :param: pos_to_find: part of speech to seek out.
    :return: dictionary of  that appear around pronouns mapped to the number of occurrences

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
    ...                   'filename': 'test_text_7.txt', 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_7.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> find_gender_pos(scarlett, "adj", common.MALE, [common.FEMALE])
    {'handsome': 3, 'sad': 1}

    """
    output = {}
    text = document.get_tokenized_text()
    if pos_to_find in pos_dict.keys():
        pos_tags = pos_dict[pos_to_find]
    else:
        #TODO this is probably bad and should be fixed.
        return "Error: invalid part of speech"

    identifiers_to_find = gender_to_find.identifiers

    identifiers_to_exclude = []
    for gender in genders_to_exclude:
        for identifier in gender.identifiers:
            identifiers_to_exclude.append(identifier)

    distances = words_instance_dist(document, identifiers_to_find)

    if len(distances) == 0:
        return {}
    elif len(distances) <= 3:
        lower_window_bound = 5
    else:
        lower_window_bound = median(sorted(distances)[:int(len(distances) / 2)])

    if not lower_window_bound >= 5:
        return "lower window bound less than 5"

    for words in windowed(text, 11):
        if not words[5].lower() in identifiers_to_find:
            continue
        if bool(set(words) & set(identifiers_to_exclude)):
            continue

        words = list(words)
        for index, word in enumerate(words):
            words[index] = word.lower()

        tags = nltk.pos_tag(words)
        for tag_index, tag in enumerate(tags):
            if tags[tag_index][1] in pos_tags:
                word = words[tag_index]
                if word in output.keys():
                    output[word] += 1
                else:
                    output[word] = 1

    return output

def find_male_pos(document, pos_to_find):
    """
    Takes in a document, returns a dictionary of poss that appear within a window of 5
    words around each male pronoun.

    :param: document
    :param: pos_to_find: the part of speech to find.
    :return: dictionary of pos that appear around male pronouns and the number of occurrences

   >>> from gender_analysis import document
   >>> from pathlib import Path
   >>> from gender_analysis import common
   >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
   ...                   'filename': 'test_text_8.txt', 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_8.txt')}
   >>> scarlett = document.Document(document_metadata)
   >>> find_male_pos(scarlett, "adj")
   {'handsome': 3, 'sad': 1}

    """

    return find_gender_pos(document, pos_to_find, common.MALE, [common.FEMALE])


def find_female_pos(document, pos_to_find):
    """
    Takes in a document, returns a dictionary of poss that appear within a window of 5
    words around each female pronoun.

    :param document: A Document object
    :param: pos_to_find: the part of speech to find.
    :return: dictionary of poss that appear around female pronouns and the number of occurrences

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
    ...                   'filename': 'test_text_9.txt', 'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_9.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> find_female_pos(scarlett, "adj")
    {'beautiful': 3, 'sad': 1}

    """

    return find_gender_pos(document, pos_to_find, common.FEMALE, [common.MALE])


def run_pos_analysis(corpus, pos_to_find, gender_list = common.BINARY_GROUP):
    """
    Takes in a corpus of novels. Return a dictionary with each novel mapped to n dictionaries, where
    n is the number of Genders in gender_list.
    The dictionary contains pos:occurences for each gendered set of identifiers.

    :param corpus: Corpus
    :param: pos_to_find: the part of speech to find.
    :return: dictionary where each key is a novel and the value is 2 dictionaries: poss and
    number of occurrences associated with male or female pronouns

    """
    results = {}

    for novel in corpus:
        results[novel] = {}
        for gender in gender_list:
            if gender.label == "Female":
                novel_result = find_female_pos(novel, pos_to_find)
            elif gender.label == "Male":
                novel_result = find_male_pos(novel, pos_to_find)
            else:
                # Note that we exclude male from female and female from male but do not do this
                # with other genders.
                novel_result = find_gender_pos(novel, pos_to_find, gender)
            if novel_result != "lower window bound less than 5":
                results[novel].update({gender.label : novel_result})

    return results

def store_raw_results(results, pickle_filepath='pronoun_pos_raw_analysis.pgz'):
    """
    Saves the results from run_pos_analysis to a pickle file.

    :param results: dictionary of results from run_pos_analysis
    :param pickle_filepath: filepath to save the output
    :return: None, saves results as pickled file with name 'pronoun_pos_raw_analysis'

    """
    try:
        common.load_pickle(pickle_filepath)
        x = input("results already stored. overwrite previous analysis? (y/n)")
        if x == 'y':
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

    >>> novel_pos_dict = {'hello': 5, 'hi': 7, 'hola': 9, 'bonjour': 2}
    >>> full_pos_dict = {'hello': 15, 'bienvenue': 3, 'hi': 23}
    >>> merge(novel_pos_dict, full_pos_dict)
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
    Takes in the full dictionary of results, returns a dictionary that maps 'male_author' and
    'female_author' to a dictionary of poss and # occurrences across novels written by
    an author of that gender.

    :param full_results: dictionary from result of run_pos_analysis
    :return: dictionary in form {'gender_author': {'male': {pos:occurrences}, 'female':{pos:occurrences}}}

    TODO: Break this binary. Easiest way is adding a 'nonbinary' category.
    """
    data = {'male_author': {'male': {}, 'female': {}}, "female_author": {'male': {}, 'female': {}}}

    for novel in list(full_results.keys()):
        author_gender = getattr(novel, 'author_gender', None)
        if author_gender == "male":
            data['male_author']['male'] = merge(full_results[novel]['male'], data['male_author']['male'])
            data['male_author']['female'] = merge(full_results[novel]['female'], data['male_author']['female'])
        elif author_gender == 'female':
            data['female_author']['male'] = merge(full_results[novel]['male'], data['female_author']['male'])
            data['female_author']['female'] = merge(full_results[novel]['female'], data['female_author']['female'])
    return data


def results_by_date(full_results, time_frame, bin_size):
    """
    Takes in a dictionary of results, returns a dictionary that maps time periods to a
    dictionary mapping poss to number of occurrences across novels written in that time period

    :param full_results: dictionary from result of run_pos_analysis
    :param time_frame: tuple (int start year, int end year) for the range of dates to return
    frequencies
    :param bin_size: int for the number of years represented in each list of frequencies
    :return: dictionary in form {date: {gender1: {pos:occurrences}, 'gender2': {pos:occurrences},
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
        for gender in genders:
            data[bin_year][gender] = merge(full_results[k][gender], data[bin_year][gender])

    return data


def results_by_location(full_results):
    """
    Takes in a dictionary of results, and returns nested dictionary that maps locations to pos
    uses by gender.

    :param full_results: dictionary from result of run_pos_analysis
    :return: dictionary in form {'location': {'gender1': {pos:occurrences},\
    'gender2':{pos:occurrences}, ...}}

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
            data[location] = gendered_output

        for gender in genders:
            data[location][gender] = merge(full_results[k][gender], data[location][gender])

    return data


def get_top_pos(full_results, num):
    """
    Takes dictionary of results from run_pos_analysis and number of top results to return.
    Returns the top num poss associated with each gender.

    :param full_results: dictionary from result of run_pos_analysis
    :param num: number of top results to return per gender
    :return: dictionary of lists of top pos associated more with each gender than the others.
    """
    merged_results = merge_raw_results(full_results)
    genders = list(merged_results.keys())

    top = {}

    excluded_results = {}
    for gender in genders:
        excluded_results[gender] = {}
        other_genders = genders.copy()
        other_genders.remove(gender)
        for pos, count in merged_results[gender].items():
            other_count = 0
            for other_gender in other_genders:
                if pos in merged_results[other_gender].keys():
                    other_count += merged_results[other_gender][pos]
            new_count = count - other_count
            excluded_results[gender][pos] = new_count

        # Sorts (pos, count) lists by count.
        top[gender] = sorted(excluded_results[gender].items(), reverse=True, key=lambda x:x[1])[0:num]

    return top
