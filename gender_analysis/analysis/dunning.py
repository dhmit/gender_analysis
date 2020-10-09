import math
from collections import Counter
from operator import itemgetter

import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from gender_analysis.corpus import Corpus
from gender_analysis.common import (
    load_graph_settings,
    MissingMetadataError,
    store_pickle,
    load_pickle,
    MASC_WORDS,
    FEM_WORDS
)


################################################################################
# BASIC DUNNING FUNCTIONS
################################################################################

def dunn_individual_word(total_words_in_corpus_1,
                         total_words_in_corpus_2,
                         count_of_word_in_corpus_1,
                         count_of_word_in_corpus_2):
    """
    applies Dunning log likelihood to compare individual word in two counter objects

    :param total_words_in_corpus_1: int, total wordcount in corpus 1
    :param total_words_in_corpus_2: int, total wordcount in corpus 2
    :param count_of_word_in_corpus_1: int, wordcount of one word in corpus 1
    :param count_of_word_in_corpus_2: int, wordcount of one word in corpus 2
    :return: Float representing the Dunning log likelihood of the given inputs

    >>> total_words_m_corpus = 8648489
    >>> total_words_f_corpus = 8700765
    >>> wordcount_female = 1000
    >>> wordcount_male = 50
    >>> dunn_individual_word(total_words_m_corpus,total_words_f_corpus,wordcount_male,wordcount_female)
    -1047.8610274053995

    """
    a = count_of_word_in_corpus_1
    b = count_of_word_in_corpus_2
    c = total_words_in_corpus_1
    d = total_words_in_corpus_2

    e1 = c * (a + b) / (c + d)
    e2 = d * (a + b) / (c + d)

    dunning_log_likelihood = 2 * (a * math.log(a / e1) + b * math.log(b / e2))

    if count_of_word_in_corpus_1 * math.log(count_of_word_in_corpus_1 / e1) < 0:
        dunning_log_likelihood = -dunning_log_likelihood

    return dunning_log_likelihood


def dunn_individual_word_by_corpus(corpus1, corpus2, word):
    """
    applies dunning log likelihood to compare individual word in two counter objects
    (-) end of spectrum is words for counter_2
    (+) end of spectrum is words for counter_1
    the larger the magnitude of the number, the more distinctive that word is in its
    respective counter object

    :param word: desired word to compare
    :param corpus1: Corpus object
    :param corpus2: Corpus object
    :return: log likelihoods and p value

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.analysis.dunning import dunn_individual_word_by_corpus
    >>> from gender_analysis.common import TEST_DATA_PATH
    >>> filepath1 = TEST_DATA_PATH / 'document_test_files'
    >>> filepath2 = TEST_DATA_PATH / 'sample_novels' / 'texts'
    >>> corpus1 = Corpus(filepath1)
    >>> corpus2 = Corpus(filepath2)
    >>> dunn_individual_word_by_corpus(corpus1, corpus2, 'sad')
    -421109.6231373814

    """

    counter1 = corpus1.get_wordcount_counter()
    counter2 = corpus2.get_wordcount_counter()

    a = counter1[word]
    b = counter2[word]
    c = 0  # total words in corpus1
    d = 0  # total words in corpus2

    for word in counter1:
        c += counter1[word]
    for word in counter2:
        d += counter2[word]

    return dunn_individual_word(a, b, c, d)


def dunning_total(counter1, counter2, pickle_filepath=None):
    """
    Runs dunning_individual on words shared by both dictionaries
    (-) end of spectrum is words for counter2
    (+) end of spectrum is words for counter1
    the larger the magnitude of the number, the more distinctive that word is in its
    respective counter object

    use pickle_filepath to store the result so it only has to be calculated once and can be
    used for multiple analyses.

    :param counter1: Python Dictionary
    :param counter2: Python Dictionary
    :param pickle_filepath: Filepath to store pickled results; will not save output if None
    :return: Dictionary

    >>> from collections import Counter
    >>> from gender_analysis.analysis.dunning import dunning_total
    >>> female_counter = Counter({'he': 1,  'she': 10, 'and': 10})
    >>> male_counter =   Counter({'he': 10, 'she': 1,  'and': 10})
    >>> results = dunning_total(female_counter, male_counter)

    Results is a dict that maps from terms to results
    Each result dict contains the dunning score...
    >>> results['he']['dunning']
    -8.547243830635558

    ... counts for corpora 1 and 2 as well as total count
    >>> results['he']['count_total'], results['he']['count_corp1'], results['he']['count_corp2']
    (11, 1, 10)

    ... and the same for frequencies
    >>> results['he']['freq_total'], results['he']['freq_corp1'], results['he']['freq_corp2']
    (0.2619047619047619, 0.047619047619047616, 0.47619047619047616)

    """

    total_words_counter1 = 0
    total_words_counter2 = 0

    # get word total in respective counters
    for word1 in counter1:
        total_words_counter1 += counter1[word1]
    for word2 in counter2:
        total_words_counter2 += counter2[word2]

    # dictionary where results will be returned
    dunning_result = {}
    for word in counter1:
        counter1_wordcount = counter1[word]
        if word in counter2:
            counter2_wordcount = counter2[word]

            if counter1_wordcount + counter2_wordcount < 10:
                continue

            dunning_word = dunn_individual_word(total_words_counter1,  total_words_counter2,
                                                counter1_wordcount,counter2_wordcount)

            dunning_result[word] = {
                'dunning': dunning_word,
                'count_total': counter1_wordcount + counter2_wordcount,
                'count_corp1': counter1_wordcount,
                'count_corp2': counter2_wordcount,
                'freq_total': (counter1_wordcount + counter2_wordcount) / (total_words_counter1 +
                                                                           total_words_counter2),
                'freq_corp1': counter1_wordcount / total_words_counter1,
                'freq_corp2': counter2_wordcount / total_words_counter2
            }

    if pickle_filepath:
        store_pickle(dunning_result, pickle_filepath)

    return dunning_result


def dunning_total_by_corpus(m_corpus, f_corpus):
    """
    Goes through two corpora, e.g. corpus of male authors and corpus of female authors
    runs dunning_individual on all words that are in BOTH corpora
    returns sorted dictionary of words and their dunning scores
    shows top 10 and lowest 10 words

    :param m_corpus: Corpus object
    :param f_corpus: Corpus object
    :return: list of tuples (common word, (dunning value, m_corpus_count, f_corpus_count))

         >>> from gender_analysis.analysis.dunning import dunning_total_by_corpus
         >>> from gender_analysis.corpus import Corpus
         >>> from gender_analysis.common import TEST_DATA_PATH
         >>> path = TEST_DATA_PATH / 'sample_novels' / 'texts'
         >>> csv_path = TEST_DATA_PATH / 'sample_novels' / 'sample_novels.csv'
         >>> c = Corpus(path, csv_path=csv_path)
         >>> m_corpus = c.filter_by_gender('male')
         >>> f_corpus = c.filter_by_gender('female')
         >>> result = dunning_total_by_corpus(m_corpus, f_corpus)
         >>> print(result[0])
         ('she', (-12374.391057010947, 29382, 45907))
         """
    wordcounter_male = m_corpus.get_wordcount_counter()
    wordcounter_female = f_corpus.get_wordcount_counter()

    totalmale_words = 0
    totalfemale_words = 0

    for male_word in wordcounter_male:
        totalmale_words += wordcounter_male[male_word]
    for female_word in wordcounter_female:
        totalfemale_words += wordcounter_female[female_word]

    dunning_result = {}
    for word in wordcounter_male:
        wordcount_male = wordcounter_male[word]
        if word in wordcounter_female:
            wordcount_female = wordcounter_female[word]

            dunning_word = dunn_individual_word(totalmale_words, totalfemale_words,
                                                wordcount_male, wordcount_female)
            dunning_result[word] = (dunning_word, wordcount_male, wordcount_female)
    dunning_result = sorted(dunning_result.items(), key=itemgetter(1))

    return dunning_result


def compare_word_association_in_corpus_dunning(word1, word2, corpus,
                                               to_pickle=False,
                                               pickle_filename='dunning_vs_associated_words.pgz'):
    """
    Uses Dunning analysis to compare the words associated with word1 vs those associated with word2 in
    the given corpus.

    :param word1: str
    :param word2: str
    :param corpus: Corpus object
    :param to_pickle: boolean; True if you wish to save the results as a Pickle file
    :param pickle_filename: str or Path object; Only used if the pickle already exists or you wish to write a new pickle file
    :return: Dictionary mapping words to dunning scores

    """
    corpus_name = corpus.name if corpus.name else 'corpus'

    try:
        results = load_pickle(pickle_filename)
    except IOError:
        try:
            pickle_filename = f'dunning_{word2}_vs_{word1}_associated_words_{corpus_name}'
            results = load_pickle(pickle_filename)
        except:
            word1_counter = Counter()
            word2_counter = Counter()
            for doc in corpus.documents:
                if isinstance(word1, str):
                    word1_counter.update(doc.words_associated(word1))
                else:  # word1 is a list of strings
                    for word in word1:
                        word1_counter.update(doc.words_associated(word))

                if isinstance(word2, str):
                    word2_counter.update(doc.words_associated(word2))
                else:  # word2 is a list of strings
                    for word in word2:
                        word2_counter.update(doc.words_associated(word))

            if to_pickle:
                results = dunning_total(word1_counter, word2_counter,
                                        pickle_filepath=pickle_filename)
            else:
                results = dunning_total(word1_counter, word2_counter)

    for group in [None, 'verbs', 'adjectives', 'pronouns', 'adverbs']:
        dunning_result_displayer(results, number_of_terms_to_display=50,
                                 part_of_speech_to_include=group)

    return results


def compare_word_association_between_corpus_dunning(word, corpus1, corpus2,
                                                    word_window=None,
                                                    to_pickle=False,
                                                    pickle_filename='dunning_associated_words.pgz'):
    """
    Finds words associated with the given word between the two corpora. The function can search the
    document automatically, or passing in a word window can refine results.

    :param word: Word to compare between the two corpora
    :param corpus1: Corpus object
    :param corpus2: Corpus object
    :param word_window: If passed in as int, trims results to only show associated words within that range.
    :param to_pickle: boolean determining if results should be pickled.
    :param pickle_filename: str or Path object, pointer to existing pickle or save location for new pickle
    :return: Dictionary

    """
    corpus1_name = corpus1.name if corpus1.name else 'corpus1'
    corpus2_name = corpus2.name if corpus2.name else 'corpus2'

    corpus1_counter = Counter()
    corpus2_counter = Counter()

    for doc in corpus1.documents:
        if word_window:
            doc.get_word_windows(word, window_size=word_window)
        else:
            if isinstance(word, str):
                corpus1_counter.update(doc.words_associated(word))
            else:  # word is a list of actual words
                for token in word:
                    corpus1_counter.update(doc.words_associated(token))

    for doc in corpus2.documents:
        if word_window:
            doc.get_word_windows(word, window_size=word_window)
        else:
            if isinstance(word, str):
                corpus2_counter.update(doc.words_associated(word))
            else:  # word is a list of actual words
                for token in word:
                    corpus2_counter.update(doc.words_associated(token))

    if to_pickle:
        results = dunning_total(corpus1_counter,
                                corpus2_counter,
                                pickle_filepath=pickle_filename)
    else:
        results = dunning_total(corpus1_counter,
                                corpus2_counter)

    for group in [None, 'verbs', 'adjectives', 'pronouns', 'adverbs']:
        dunning_result_displayer(results, number_of_terms_to_display=20,
                                 corpus1_display_name=f'{corpus1_name}. {word}',
                                 corpus2_display_name=f'{corpus2_name}. {word}',
                                 part_of_speech_to_include=group)

    return results


def dunning_result_to_dict(dunning_result,
                           number_of_terms_to_display=10,
                           part_of_speech_to_include=None):
    """
    Receives a dictionary of results and returns a dictionary of the top
    number_of_terms_to_display most distinctive results for each corpus that have a part of speech
    matching part_of_speech_to_include

    :param dunning_result: Dunning result dict that will be sorted through
    :param number_of_terms_to_display: Number of terms for each corpus to display
    :param part_of_speech_to_include: 'adjectives', 'adverbs', 'verbs', or 'pronouns'
    :return: dict

    """

    pos_names_to_tags = {
        'adjectives': ['JJ', 'JJR', 'JJS'],
        'adverbs': ['RB', 'RBR', 'RBS', 'WRB'],
        'verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'pronouns': ['PRP', 'PRP$', 'WP', 'WP$']
    }
    if part_of_speech_to_include in pos_names_to_tags:
        part_of_speech_to_include = pos_names_to_tags[part_of_speech_to_include]

    final_results_dict = {}

    reverse = True
    for i in range(2):
        sorted_results = sorted(dunning_result.items(), key=lambda x: x[1]['dunning'],
                                    reverse=reverse)
        count_displayed = 0
        for result in sorted_results:
            if count_displayed == number_of_terms_to_display:
                break
            term = result[0]
            term_pos = nltk.pos_tag([term])[0][1]
            if part_of_speech_to_include and term_pos not in part_of_speech_to_include:
                continue

            final_results_dict[result[0]] = result[1]
            count_displayed += 1
        reverse = False
    return final_results_dict


################################################################################
# Visualizers
################################################################################

def dunning_result_displayer(dunning_result, number_of_terms_to_display=10,
                             corpus1_display_name=None, corpus2_display_name=None,
                             part_of_speech_to_include=None, save_to_filename=None):
    """
    Convenience function to display dunning results as tables.

    part_of_speech_to_include can either be a list of POS tags or a 'adjectives, 'adverbs',
    'verbs', or 'pronouns'. If it is None, all terms are included.

    Optionally save the output to a text file

    :param dunning_result:              Dunning result dict to display
    :param number_of_terms_to_display:  Number of terms for each corpus to display
    :param corpus1_display_name:        Name of corpus 1 (e.g. "Female Authors")
    :param corpus2_display_name:        Name of corpus 2 (e.g. "Male Authors")
    :param part_of_speech_to_include:   e.g. 'adjectives', or 'verbs'
    :param save_to_filename:            Filename to save output
    :return:
    """

    pos_names_to_tags = {
        'adjectives':   ['JJ', 'JJR', 'JJS'],
        'adverbs':      ['RB', 'RBR', 'RBS', 'WRB'],
        'verbs':        ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'pronouns':     ['PRP', 'PRP$', 'WP', 'WP$']
    }
    if part_of_speech_to_include in pos_names_to_tags:
        part_of_speech_to_include = pos_names_to_tags[part_of_speech_to_include]

    if not corpus1_display_name:
        corpus1_display_name = 'Corpus 1'
    if not corpus2_display_name:
        corpus2_display_name = 'Corpus 2'

    headings = ['term', 'dunning', 'count_total', 'count_corp1', 'count_corp2', 'freq_total',
                'freq_corp1', 'freq_corp2']

    output = f'\nDisplaying Part of Speech: {part_of_speech_to_include}\n'
    for i, name in enumerate([corpus1_display_name, corpus2_display_name]):
        output += f'\nDunning Log-Likelihood results for {name}\n|'

        for heading in headings:
            heading = heading.replace('_corp1', ' ' + corpus1_display_name).replace('_corp2',
                                                                       ' ' + corpus2_display_name)
            output += ' {:19s}|'.format(heading)
        output += '\n' + 8 * 21 * '_' + '\n'

        reverse = True
        if i == 1:
            reverse = False
        sorted_results = sorted(dunning_result.items(), key=lambda x: x[1]['dunning'],
                                reverse=reverse)
        count_displayed = 0
        for result in sorted_results:
            if count_displayed == number_of_terms_to_display:
                break
            term = result[0]
            term_pos = nltk.pos_tag([term])[0][1]
            if part_of_speech_to_include and term_pos not in part_of_speech_to_include:
                continue

            output += '|  {:18s}|'.format(result[0])
            for heading in headings[1:]:

                if heading in ['freq_total', 'freq_corp1', 'freq_corp2']:
                    output += '  {:16.4f}% |'.format(result[1][heading] * 100)
                elif heading in ['dunning']:
                    output += '  {:17.2f} |'.format(result[1][heading])
                else:
                    output += '  {:17.0f} |'.format(result[1][heading])
            output += '\n'
            count_displayed += 1

    if save_to_filename:
        with open(save_to_filename + '.txt', 'w') as outfile:
            outfile.write(output)

    print(output)


def score_plot_to_show(results):
    """
    displays bar plot of dunning scores for all words in results

    :param results: dict of results from dunning_total or similar, i.e. in the form {'word': {
        'dunning': float}}
    :return: None, displays bar plot of dunning scores for all words in results

    """
    load_graph_settings(False)
    results_dict = dict(results)
    words = []
    dunning_score = []

    for term, data in results_dict.items():
        words.append(term)
        dunning_score.append(data['dunning'])

    opacity = 0.4

    colors = ['r' if entry >= 0 else 'b' for entry in dunning_score]
    ax = sns.barplot(dunning_score, words, palette=colors, alpha=opacity)
    sns.despine(ax=ax, bottom=True, left=True)
    plt.show()


def freq_plot_to_show(results):
    """
    displays bar plot of relative frequency of all words in results

    :param results: dict of results from dunning_total or similar, i.e. in the form {'word': {
        'freq_corp1': int, 'freq_corp2': int, 'freq_total': int}}
    :return: None, displays bar plot of relative frequency of all words in results

    """
    load_graph_settings(False)
    results_dict = dict(results)
    words = []
    female_rel_freq = []
    male_rel_freq = []

    for term, data in results_dict.items():
        words.append(term)
        female_rel_freq.append(data['freq_corp1']/data['freq_total'])
        male_rel_freq.append(-1*data['freq_corp2']/data['freq_total'])

    opacity = 0.4

    colors = ['b']
    ax = sns.barplot(male_rel_freq, words, palette=colors, alpha=opacity)
    sns.despine(ax=ax, bottom=True, left=True)
    plt.show()


################################################################################
# Individual Analyses
################################################################################

# Words associated with male and female characters,
# based on whether author is male or female

def male_characters_author_gender_differences(corpus, to_pickle=False,
                                        pickle_filename='dunning_male_chars_author_gender.pgz'):
    """
    Between male-author and female-author subcorpora, tests distinctiveness of words associated
    with male characters

    Prints out the most distinctive terms overall as well as grouped by verbs, adjectives etc.

    :param corpus: Corpus object
    :param to_pickle: boolean, False by default. Set to True in order to pickle results
    :param pickle_filename: filename of results to be pickled
    :return: dict

    """

    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    m_corpus = corpus.filter_by_gender('male')
    f_corpus = corpus.filter_by_gender('female')

    return compare_word_association_between_corpus_dunning(MASC_WORDS, m_corpus, f_corpus,
                                                           word_window=None, to_pickle=to_pickle,
                                                           pickle_filename=pickle_filename)


def female_characters_author_gender_differences(corpus,
                                                to_pickle=False,
                                                pickle_filename='dunning_female_chars_author_gender.pgz'):
    """
    Between male-author and female-author subcorpora, tests distinctiveness of words associated
    with male characters

    Prints out the most distinctive terms overall as well as grouped by verbs, adjectives etc.

    :param corpus: Corpus object
    :param to_pickle: boolean, False by default. Set to True in order to pickle results
    :param pickle_filename: filename of results to be pickled
    :return: dict

    """

    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    m_corpus = corpus.filter_by_gender('male')
    f_corpus = corpus.filter_by_gender('female')

    return compare_word_association_between_corpus_dunning(FEM_WORDS, m_corpus, f_corpus,
                                                           word_window=None, to_pickle=to_pickle,
                                                           pickle_filename=pickle_filename)


def dunning_words_by_author_gender(corpus, display_results=False, to_pickle=False,
                                   pickle_filename='dunning_male_vs_female_authors.pgz'):
    """
    Tests distinctiveness of shared words between male and female authors using dunning analysis.

    If called with display_results=True, prints out the most distinctive terms overall as well as
    grouped by verbs, adjectives etc.
    Returns a dict of all terms in the corpus mapped to the dunning data for each term

    :param corpus: Corpus object
    :param display_results: Boolean; reports a visualization of the results if True
    :param to_pickle: Boolean; Will save the results to a pickle file if True
    :param pickle_filename: Path to pickle object; will try to search for results in this location or write pickle file to path if to_pickle is true.
    :return: dict

    """

    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    # By default, try to load precomputed results. Only calculate if no stored results are
    # available.
    try:
        results = load_pickle(pickle_filename)
    except IOError:

        m_corpus = corpus.filter_by_gender('male')
        f_corpus = corpus.filter_by_gender('female')
        wordcounter_male = m_corpus.get_wordcount_counter()
        wordcounter_female = f_corpus.get_wordcount_counter()
        if to_pickle:
            results = dunning_total(wordcounter_female, wordcounter_male,
                                    filename_to_pickle=pickle_filename)
        else:
            results = dunning_total(wordcounter_female, wordcounter_male)

    if display_results:
        for group in [None, 'verbs', 'adjectives', 'pronouns', 'adverbs']:
            dunning_result_displayer(results, number_of_terms_to_display=20,
                                     corpus1_display_name='Fem Author',
                                     corpus2_display_name='Male Author',
                                     part_of_speech_to_include=group)
    return results


def masc_fem_associations_dunning(corpus,
                                  to_pickle=False,
                                  pickle_filename='dunning_he_vs_she_associated_words.pgz'):
    """
    Uses Dunning analysis to compare words associated with MASC_WORDS vs. words associated
    with FEM_WORDS in a given Corpus.

    :param corpus: Corpus object
    :param to_pickle: Boolean; saves results to a pickle file if True
    :param pickle_filename: Filepath to save pickle file if to_pickle is True
    :return: Dictionary

    """
    if to_pickle:
        return compare_word_association_in_corpus_dunning(MASC_WORDS, FEM_WORDS, corpus,
                                                          to_pickle=True,
                                                          pickle_filename=pickle_filename)
    return compare_word_association_in_corpus_dunning(MASC_WORDS, FEM_WORDS, corpus)


