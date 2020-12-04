from gender_analysis import common
from gender_analysis.analysis import gender_pos


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

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': \
    '1966', 'filename': 'test_text_7.txt', 'filepath': Path(common.TEST_DATA_PATH, \
    'document_test_files', 'test_text_7.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> find_gender_adj(scarlett, common.MALE, genders_to_exclude=[common.FEMALE])
    {'handsome': 3, 'sad': 1}

    """

    return gender_pos.find_gender_pos(document, 'adj',
                                      gender_to_find, word_window, genders_to_exclude)


def find_male_adj(document):
    """
    Takes in a document, returns a dictionary of adjectives that appear within a window of 5
    words around each male pronoun.

    :param: document
    :return: dictionary of adjectives that appear around male pronouns and the number of occurrences

   >>> from gender_analysis import document
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

    >>> from gender_analysis import document
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

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
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


def run_adj_analysis_doc(document, gender_list=common.BINARY_GROUP):
    """
    Takes in a document and a list of genders to analyze,
    returns a dictionary with the find_gender_adj results for each gender in gender_list.

    :param document: a Document object
    :param gender_list: a list of genders to run the adjective search for.
    :return: a dictionary of find_gender_adj results for each gender in gender_list.

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.analysis.gender_adjective import run_adj_analysis_doc
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings = True)
    >>> flatland = corpus.get_document("title", "Flatland")
    >>> adj_dict = run_adj_analysis_doc(flatland)
    >>> adj_dict['Female']['invisible']
    3

    """

    return gender_pos.run_pos_analysis_doc(document, 'adj', gender_list)


def store_raw_results(results, pickle_filepath='pronoun_adj_raw_analysis.pgz'):
    """
    Saves the results from run_adj_analysis to a pickle file.

    :param results: dictionary of results from run_adj_analysis
    :param pickle_filepath: filepath to save the output
    :return: None, saves results as pickled file with name 'pronoun_adj_raw_analysis'

    """

    gender_pos.store_raw_results(results, pickle_filepath)


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

    return gender_pos.merge(novel_adj_dict, full_adj_dict)


def merge_raw_results(full_results):
    """
    Merges all adjectives across novels into dictionaries sorted by gender.

    :param full_results: full corpus results from run_adj_analysis
    :return: dictionary in the form {'gender':{'adj': occurrences}}
    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> from gender_analysis.analysis.gender_adjective import run_adj_analysis
    >>> tiny_corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> tiny_results = run_adj_analysis(tiny_corpus)
    >>> tiny_merged = merge_raw_results(tiny_results)
    >>> tiny_merged['Male']['strong']
    4
    """

    return gender_pos.merge_raw_results(full_results)


def get_overlapping_adjectives_raw_results(merged_results):
    """
    Looks through the gendered adjectives across the corpus and extracts adjectives that overlap
    that overlap across all genders and their occurrences.
    FORMAT - {'adjective': [gender1, gender2, ...]}
    :param merged_results: the results of merge_raw_results.
    :return: dict in form {'adjective': ['gender1': occurrences, 'gender2': occurrences, ... }}

    TODO: Consider adding a more granular function that gets all sub-intersections.
    """

    return gender_pos.get_overlapping_pos_raw_results(merged_results)


def results_by_author_gender(full_results):
    """
    Takes in a dictionary of results, and returns nested dictionary that maps author_gender to
    adjective uses by gender.

    :param full_results: dictionary from result of run_adj_analysis
    :return: dictionary in form {'author_gender': {'gender1': {adj:occurrences},\
    'gender2':{adj:occurrences}, ...}}

    """

    return gender_pos.results_by_author_gender(full_results)


def results_by_location(full_results):
    """
    Takes in a dictionary of results, and returns nested dictionary that maps locations to adjective
    uses by gender.

    :param full_results: dictionary from result of run_adj_analysis
    :return: dictionary in form {'location': {'gender1': {adj:occurrences},\
    'gender2':{adj:occurrences}, ...}}
    """

    return gender_pos.results_by_location(full_results)


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

    return gender_pos.get_top_pos(full_results, num, remove_swords)


def display_binned_results(metadata_binned_results, num_to_return, remove_swords=False):
    """
    Takes in the results of results_by_(metadata_field) and returns a reader-friendly set of dicts.

    :param metadata_binned_results: The results of results_by_author_gender, location, or date.
    :param num_to_return: Number of adjectives to return
    :param remove_swords: Whether or not to remove English stopwords
    :return: A formatted, sorted dictionary of results.
    """

    return gender_pos.display_binned_results(metadata_binned_results, num_to_return,
                                             remove_swords)


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

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH, TINY_TEST_CORPUS_CSV
    >>> tiny_corpus = Corpus(TEST_CORPUS_PATH, csv_path=TINY_TEST_CORPUS_CSV, ignore_warnings=True)
    >>> flatland = tiny_corpus.get_document("title", "Flatland")
    >>> flatland_adjs = run_adj_analysis_doc(flatland)
    >>> flatland_diffs = difference_adjs(flatland_adjs)
    >>> flatland_diffs['Female'][0]
    ('invisible', 3)
    """

    return gender_pos.difference_pos(gender_adj_dict, num_to_return)


def display_gender_adjectives(result_dict, num_to_return=10, remove_stopwords=True):
    """
    takes the results of find_gender_adj and prints out to the user the top number_to_return
    adjectives associated with the gender searched for, sorted

    :param result_dict:a dict of word-value frequency
    :param num_to_return: top num of words to be returned to the user -> threshold, or the top?
    :param remove_stopwords: removes English stopwords
    :return: tuples of sorted top num_to_return words with their freq
    """

    return gender_pos.display_gender_pos(result_dict, num_to_return, remove_stopwords)
