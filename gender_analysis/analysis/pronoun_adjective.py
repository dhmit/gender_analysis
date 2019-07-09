from statistics import median
import nltk
from more_itertools import windowed

from gender_analysis.analysis.instance_distance import male_instance_dist, female_instance_dist
from gender_analysis import common


def find_gender_adj(document, female):
    """
        Takes in a document and boolean indicating gender, returns a dictionary of adjectives that
        appear within a window of 5 words around each pronoun
        >>> from gender_analysis import document
        >>> from pathlib import Path
        >>> from gender_analysis import common
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
        ...                   'filename': 'test_text_7.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'document_test_files', 'test_text_7.txt')}
        >>> scarlett = document.Document(document_metadata)
        >>> find_gender_adj(scarlett, False)
        {'handsome': 3, 'sad': 1}

        :param: document: Document
        :param: female: boolean indicating whether to search for female adjectives (true) or
        male adjectives (false)
        :return: dictionary of adjectives that appear around male pronouns mapped to the number of
        occurrences
    """
    output = {}
    text = document.get_tokenized_text()

    if female:
        distances = female_instance_dist(document)
        pronouns1 = ["her", "hers", "she", "herself"]
        pronouns2 = ["his", "him", "he", "himself"]
    else:
        distances = male_instance_dist(document)
        pronouns1 = ["his", "him", "he", "himself"]
        pronouns2 = ["her", "hers", "she", "herself"]
    if len(distances) == 0:
        return {}
    elif len(distances) <= 3:
        lower_window_bound = 5
    else:
        lower_window_bound = median(sorted(distances)[:int(len(distances) / 2)])

    if not lower_window_bound >= 5:
        return "lower window bound less than 5"
    for l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11 in windowed(text, 11):
        l6 = l6.lower()
        if not l6 in pronouns1:
            continue
        words = [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11]
        if bool(set(words) & set(pronouns2)):
            continue
        for index, word in enumerate(words):
            words[index] = word.lower()
        tags = nltk.pos_tag(words)
        for tag_index, tag in enumerate(tags):
            if tags[tag_index][1] == "JJ" or tags[tag_index][1] == "JJR" or tags[tag_index][1] == "JJS":
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
       >>> from gender_analysis import document
       >>> from pathlib import Path
       >>> from gender_analysis import common
       >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
       ...                   'filename': 'test_text_8.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'document_test_files', 'test_text_8.txt')}
       >>> scarlett = document.Document(document_metadata)
       >>> find_male_adj(scarlett)
       {'handsome': 3, 'sad': 1}

       :param:document
       :return: dictionary of adjectives that appear around male pronouns and the number of
       occurrences
    """
    return find_gender_adj(document, False)


def find_female_adj(document):
    """
        Takes in a document, returns a dictionary of adjectives that appear within a window of 5
        words around each female pronoun
       >>> from gender_analysis import document
       >>> from pathlib import Path
       >>> from gender_analysis import common
       >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1966',
       ...                   'filename': 'test_text_9.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'document_test_files', 'test_text_9.txt')}
       >>> scarlett = document.Document(document_metadata)
       >>> find_female_adj(scarlett)
       {'beautiful': 3, 'sad': 1}

       :param:document
       :return: dictionary of adjectives that appear around female pronouns and the number of
       occurrences

       """
    return find_gender_adj(document, True)


def run_adj_analysis(corpus):
    """
    Takes in a corpus of novels. Return a dictionary with each novel mapped to 2 dictionaries: adjectives/#occurences
    associated with male pronouns, and adjectives/# occurrences associated with female pronouns

    :param corpus: Corpus
    :return:dictionary where each key is a novel and the value is 2 dictionaries:
    - Adjectives and number of occurrences associated with male pronouns
    - Adjectives and number of occurrences associated with female pronouns

    """
    results = {}
    for novel in corpus:
        novel_male_results = find_male_adj(novel)
        novel_female_results = find_female_adj(novel)
        if (novel_male_results != "lower window bound less than 5"
                and novel_female_results != "lower window bound less than 5"):
            novel.text = ""
            novel._word_counts_counter = None
            results[novel] = {'male': novel_male_results, 'female': novel_female_results}

    return results


def store_raw_results(results, pickle_filepath='pronoun_adj_raw_analysis.pgz'):
    """
    :param results: dictionary of results from run_adj_analysis
    :param pickle_filepath: filepath to save the output
    :return: None, saves results as pickled file with name 'pronoun_adj_raw_analysis'
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


def merge(novel_adj_dict, full_adj_dict):
    """
    Merges adjective occurrence results from a single novel with results for each novel.
    :param novel_adj_dict: dictionary of adjectives/#occurrences for one novel
    :param full_adj_dict: dictionary of adjectives/#occurrences for multiple novels
    :return: full_results dictionary with novel_results merged in

        >>> novel_adj_dict = {'hello': 5, 'hi': 7, 'hola': 9, 'bonjour': 2}
        >>> full_adj_dict = {'hello': 15, 'bienvenue': 3, 'hi': 23}
        >>> merge(novel_adj_dict, full_adj_dict)
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
    Merges all of the male adjectives across novels into a single dictionary, and merges all of the female adjectives
    across novels into a single dictionary.
    :param full_results: full corpus results from run_adj_analysis
    :return: dictionary in the form {'gender':{'adj': occurrences}}
    """
    merged_results = {'male': {}, 'female': {}}
    for novel in list(full_results.keys()):
        print(novel.title, novel.author)
        for gender in list(full_results[novel].keys()):
            merged_results[gender] = merge(full_results[novel][gender], merged_results[gender])

    return merged_results


def get_overlapping_adjectives_raw_results(merged_results):
    """
    Looks through the male adjectives and female adjectives across the corpus and extracts adjective
    that overlap across both and their occurrences. FORMAT - {'adjective': [male, female]}
    :param merged_results:
    :return: dictionary in the form {'adjective':{'gender': num occurrences associated with gender}}
    """
    overlap_results = {}
    male_adj = list(merged_results['male'].keys())
    female_adj = list(merged_results['female'].keys())

    for a in male_adj:
        if a in female_adj:
            overlap_results[a] = [merged_results['male'][a], merged_results['female'][a]]

    return overlap_results


def results_by_author_gender(full_results):
    """
       takes in the full dictionary of results, returns a dictionary that maps 'male_author' and
       'female_author' to a dictionary of adjectives and # occurrences across novels written by
       an author of that gender.
       :param full_results: dictionary from result of run_adj_analysis
       :return: dictionary in form {'gender_author': {'male': {adj:occurrences}, 'female':{
       adj:occurrences}}}
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
    takes in a dictionary of results, returns a dictionary that maps time periods to a
    dictionary mapping adjectives to number of occurrences across novels written in that time period

    :param full_results: dictionary from result of run_adj_analysis
    :param time_frame: tuple (int start year, int end year) for the range of dates to return
    frequencies
    :param bin_size: int for the number of years represented in each list of frequencies
    :return: dictionary in form {date: {'male': {adj:occurrences}, 'female':{adj:occurrences}}},
    where date is the first year in its bin
    """
    data = {}
    for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
        data[bin_start_year] = {'male': {}, 'female': {}}

    for k in full_results.keys():
        date = getattr(k, 'date', None)
        if date is None:
            continue
        bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
        data[bin_year]['male'] = merge(full_results[k]['male'], data[bin_year]['male'])
        data[bin_year]['female'] = merge(full_results[k]['female'], data[bin_year['female']])

    return data


def results_by_location(full_results):
    """
    :param full_results: dictionary from result of run_adj_analysis
    :return: dictionary in form {'location': {'male': {adj:occurrences}, 'female':{adj:occurrences}}}
    """
    data = {}

    for k in full_results.keys():
        location = getattr(k, 'country_publication', None)
        if location is None:
            continue

        if location not in data:
            data[location] = {'male': {}, 'female': {}}
        data[location]['male'] = merge(full_results[k]['male'], data[location]['male'])
        data[location]['female'] = merge(full_results[k]['female'], data[location]['female'])

    return data


def get_top_adj(full_results, num):
    """
    Takes dictionary of results from run_adj_analysis and number of top results to return.
    Returns the top num adjectives associated with male pronouns and female pronouns
    :param full_results: dictionary from result of run_adj_analysis
    :param num: number of top results to return per gender
    :return: tuple of lists of top adjectives associated with male pronouns and female pronouns,
    respectively
    """
    male_adj = []
    female_adj = []

    for adj, val in full_results.items():
        male_adj.append((val[0]-val[1], adj))
        female_adj.append((val[1]-val[0], adj))

    male_top = sorted(male_adj, reverse=True)[0:num]
    female_top = sorted(female_adj, reverse=True)[0:num]

    return male_top, female_top
