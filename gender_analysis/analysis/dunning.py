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
    load_pickle
)


# TODO: Rewrite all of this using a Dunning class in a non-messy way.


def dunn_individual_word(total_words_in_corpus_1, total_words_in_corpus_2,
                         count_of_word_in_corpus_1,
                         count_of_word_in_corpus_2):
    """
    applies Dunning log likelihood to compare individual word in two counter objects

    :param total_words_in_corpus_1: int, total wordcount in corpus 1
    :param total_words_in_corpus_2: int, total wordcount in corpus 2
    :param count_of_word_in_corpus_1: int, wordcount of one word in corpus 1
    :param count_of_word_in_corpus_2: int, wordcount of one word in corpus 2
    :return: Dunning log likelihood
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
    :param corpus1: Corpus
    :param corpus2: Corpus
    :return: log likelihoods and p value
    # TODO: fix doctest for new corpus input
    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.analysis.dunning import dunn_individual_word_by_corpus
    >>> from gender_analysis.common import BASE_PATH
    >>> filepath1 = BASE_PATH / 'testing' / 'corpora' / 'document_test_files'
    >>> filepath2 = BASE_PATH / 'testing' / 'corpora' / 'sample_novels' / 'texts'
    >>> corpus1 = Corpus(filepath1)
    >>> corpus2 = Corpus(filepath2)
    >>> dunn_individual_word_by_corpus(corpus1, corpus2, 'sad')
    -425133.12886726425
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
    runs dunning_individual on words shared by both counter objects
    (-) end of spectrum is words for counter_2
    (+) end of spectrum is words for counter_1
    the larger the magnitude of the number, the more distinctive that word is in its
    respective counter object

    use filename_to_pickle to store the result so it only has to be calculated once and can be
    used for multiple analyses.

    >>> from collections import Counter
    >>> from gender_analysis.analysis.dunning import dunning_total
    >>> female_counter = Counter({'he': 1,  'she': 10, 'and': 10})
    >>> male_counter =   Counter({'he': 10, 'she': 1,  'and': 10})
    >>> results = dunning_total(female_counter, male_counter)

    # Results is a dict that maps from terms to results
    # Each result dict contains the dunning score...
    >>> results['he']['dunning']
    -8.547243830635558

    # ... counts for corpora 1 and 2 as well as total count
    >>> results['he']['count_total'], results['he']['count_corp1'], results['he']['count_corp2']
    (11, 1, 10)

    # ... and the same for frequencies
    >>> results['he']['freq_total'], results['he']['freq_corp1'], results['he']['freq_corp2']
    (0.2619047619047619, 0.047619047619047616, 0.47619047619047616)

    :return: dict

    """

    total_words_counter1 = 0
    total_words_counter2 = 0

    # get word total in respective counters
    for word1 in counter1:
        total_words_counter1 += counter1[word1]
    for word2 in  counter2:
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
    goes through two corpora, e.g. corpus of male authors and corpus of female authors
    runs dunning_individual on all words that are in BOTH corpora
    returns sorted dictionary of words and their dunning scores
    shows top 10 and lowest 10 words

    :param m_corpus: Corpus
    :param f_corpus: Corpus

    :return: list of tuples (common word, (dunning value, m_corpus_count, f_corpus_count))

         >>> from gender_analysis.analysis.analysis import dunning_total
         >>> from gender_analysis.corpus import Corpus
         >>> from gender_analysis.common import BASE_PATH
         >>> path = BASE_PATH / 'testing' / 'corpora' / 'sample_novels' / 'texts'
         >>> csv_path = BASE_PATH / 'testing' / 'corpora' / 'sample_novels' / 'sample_novels.csv'
         >>> c = Corpus(path, csv_path=csv_path)
         >>> m_corpus = c.filter_by_gender('male')
         >>> f_corpus = c.filter_by_gender('female')
         >>> result = dunning_total(m_corpus, f_corpus)
         >>> print(result[0])
         ('she', (-12320.96452787667, 29100, 45548))
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


def male_vs_female_authors_analysis_dunning_lesser(corpus):
    """
    tests word distinctiveness of shared words between male and female corpora using dunning
    :return: dictionary of common shared words and their distinctiveness
    """

    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    m_corpus = corpus.filter_by_gender('male')
    f_corpus = corpus.filter_by_gender('female')
    wordcounter_male = m_corpus.get_wordcount_counter()
    wordcounter_female = f_corpus.get_wordcount_counter()
    results = dunning_total(wordcounter_male, wordcounter_female)
    list_results = list(results.keys())
    list_results.sort(key=lambda x: results[x]['dunning'])
    print("women's top 10: ", list_results[0:10])
    print("men's top 10: ", list(reversed(list_results[-10:])))
    return results

    
def dunning_result_displayer(dunning_result, number_of_terms_to_display=10,
                             corpus1_display_name=None, corpus2_display_name=None,
                             part_of_speech_to_include=None):
    """
    Convenience function to display dunning results as tables.

    part_of_speech_to_include can either be a list of POS tags or a 'adjectives, 'adverbs',
    'verbs', or 'pronouns'. If it is None, all terms are included.

    :param dunning_result:              Dunning result dict to display
    :param number_of_terms_to_display:  Number of terms for each corpus to display
    :param corpus1_display_name:        Name of corpus 1 (e.g. "Female Authors")
    :param corpus2_display_name:        Name of corpus 2 (e.g. "Male Authors")
    :param part_of_speech_to_include:   e.g. 'adjectives', or 'verbs'
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

    print(output)


def compare_word_association_in_corpus_analysis_dunning(word1, word2, corpus, to_pickle=False, pickle_filename='dunning_vs_associated_words.pgz'):
    """
    Uses Dunning analysis to compare words associated with word1 vs words associated with word2 in
    the Corpus passed in as the parameter.
    :param word1: str
    :param word2: str
    :param corpus: Corpus
    :param to_pickle: boolean
    :return: dict
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
                word1_counter.update(doc.words_associated(word1))
                word2_counter.update(doc.words_associated(word2))
            if to_pickle:
                results = dunning_total(word1_counter, word2_counter,
                                        filename_to_pickle=pickle_filename)
            else:
                results = dunning_total(word1_counter, word2_counter)

    for group in [None, 'verbs', 'adjectives', 'pronouns', 'adverbs']:
        dunning_result_displayer(results, number_of_terms_to_display=50,
                                 part_of_speech_to_include=group)

    return results


def compare_word_association_between_corpus_analysis_dunning(word, corpus1, corpus2,
                                                             word_window=None,
                                                             to_pickle=False,
                                                             pickle_filename='dunning_associated_words.pgz'):
    """
    Uses Dunning analysis to compare words associated with word between corpuses.

    :param word: str
    :param corpus1: Corpus
    :param corpus2: Corpus
    :param word_window
    :param to_pickle: boolean determining if results should be pickled
    :return: dict
    """
    corpus1_name = corpus1.name if corpus1.name else 'corpus1'
    corpus2_name = corpus2.name if corpus2.name else 'corpus2'

    if word_window:
        pickle_filename += f'_word_window_{word_window}'
    try:
        results = load_pickle(pickle_filename)
    except IOError:
        print("Precalculated result not available. Running analysis now...")
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
            results = dunning_total(corpus1_counter, corpus2_counter,
                                    pickle_filepath=pickle_filename)
        else:
            results = dunning_total(corpus1_counter, corpus2_counter)

    for group in [None, 'verbs', 'adjectives', 'pronouns', 'adverbs']:
        dunning_result_displayer(results, number_of_terms_to_display=20,
                                 corpus1_display_name=f'{corpus1_name}. {word}',
                                 corpus2_display_name=f'{corpus2_name}. {word}',
                                 part_of_speech_to_include=group)

    return results


def male_vs_female_analysis_dunning(corpus, display_data=False, to_pickle=False, pickle_filename='dunning_male_vs_female_chars.pgz'):
    """
    tests word distinctiveness of shared words between male and female corpora using dunning
    Prints out the most distinctive terms overall as well as grouped by verbs, adjectives etc.

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

        from collections import Counter
        wordcounter_male = Counter()
        wordcounter_female = Counter()

        for novel in m_corpus:
            wordcounter_male += novel.words_associated('he')

        for novel in f_corpus:
            wordcounter_female += novel.words_associated('he')

        if to_pickle:
            results = dunning_total(wordcounter_male, wordcounter_female,
                                    filename_to_pickle=pickle_filename)
        else:
            results = dunning_total(wordcounter_male, wordcounter_female)
    if display_data:
        for group in [None, 'verbs', 'adjectives', 'pronouns', 'adverbs']:
            dunning_result_displayer(results, number_of_terms_to_display=20,
                                     corpus1_display_name='Fem Author',
                                     corpus2_display_name='Male Author',
                                     part_of_speech_to_include=group)
    return results


def dunning_result_to_dict(dunning_result, number_of_terms_to_display=10,
                             part_of_speech_to_include=None):
    """
    Receives a dictionary of results and returns a dictionary of the top
    number_of_terms_to_display most distinctive results for each corpus that have a part of speech
    matching part_of_speech_to_include
    :param dunning_result:              Dunning result dict that will be sorted through
    :param number_of_terms_to_display:  Number of terms for each corpus to display
    :param part_of_speech_to_include:   e.g. 'adjectives', or 'verbs'
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


################################################
# Individual Analyses                          #
################################################


# Male Authors versus Female Authors
################################################

def male_vs_female_authors_analysis_dunning(corpus, display_results=False, to_pickle=False, pickle_filename='dunning_male_vs_female_authors.pgz'):
    """
    tests word distinctiveness of shared words between male and female authors using dunning
    If called with display_results=True, prints out the most distinctive terms overall as well as
    grouped by verbs, adjectives etc.
    Returns a dict of all terms in the corpus mapped to the dunning data for each term

    :return:dict
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


# Male Characters versus Female Characters (words following 'he' versus words following 'she')
##############################################################################################

def he_vs_she_associations_analysis_dunning(corpus, to_pickle=False, pickle_filename='dunning_he_vs_she_associated_words.pgz'):
    """
    Uses Dunning analysis to compare words associated with 'he' vs words associated with 'she' in
    the Corpus passed in as the parameter.
    :param corpus: Corpus
    :param to_pickle: boolean
    """

    try:
        results = load_pickle(pickle_filename)
    except IOError:
        he_counter = Counter()
        she_counter = Counter()
        for doc in corpus.documents:
            he_counter.update(doc.words_associated("he"))
            she_counter.update(doc.words_associated("she"))
        if to_pickle:
            results = dunning_total(she_counter, he_counter, filename_to_pickle=pickle_filename)
        else:
            results = dunning_total(she_counter, he_counter)

    for group in [None, 'verbs', 'adjectives', 'pronouns', 'adverbs']:
        dunning_result_displayer(results, number_of_terms_to_display=20,
                                 corpus1_display_name='she...',
                                 corpus2_display_name='he..',
                                 part_of_speech_to_include=group)


# Female characters as written by Male Authors versus Female Authors
####################################################################

def female_characters_author_gender_differences(corpus, to_pickle=False):
    """
    Compares how male authors versus female authors write female characters by looking at the words
    that follow 'she'

    :param corpus: Corpus
    :param to_pickle
    :return:
    """

    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    male_corpus = corpus.filter_by_gender('male')
    female_corpus = corpus.filter_by_gender('female')
    return compare_word_association_between_corpus_analysis_dunning(word='she',
                                                                    corpus1=female_corpus, corpus2=male_corpus,
                                                                    to_pickle=to_pickle)


# Male characters as written by Male Authors versus Female Authors
####################################################################

def male_characters_author_gender_differences(corpus, to_pickle=False):
    """
    Compares how male authors versus female authors write male characters by looking at the words
    that follow 'he'

    :param corpus: Corpus
    :param to_pickle
    :return:
    """
    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    male_corpus = corpus.filter_by_gender('male')
    female_corpus = corpus.filter_by_gender('female')
    return compare_word_association_between_corpus_analysis_dunning(word='he',
                                                                    corpus1=female_corpus, corpus2=male_corpus,
                                                                    to_pickle=to_pickle)


# God as written by Male Authors versus Female Authors
####################################################################

def god_author_gender_differences(corpus, to_pickle=False):
    """
    Compares how male authors versus female authors refer to God by looking at the words
    that follow 'God'

    :param corpus: Corpus
    :param to_pickle
    :return:
    """
    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    male_corpus = corpus.filter_by_gender('male')
    female_corpus = corpus.filter_by_gender('female')
    return compare_word_association_between_corpus_analysis_dunning(word='God',
                                                                  corpus1=female_corpus, corpus2=male_corpus,
                                                                    to_pickle=to_pickle)


def money_author_gender_differences(corpus, to_pickle=False):
    """
    Compares how male authors versus female authors refer to money by looking at the words
   before and after money'

    :param corpus: Corpus
    :param to_pickle
    :return:
    """
    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    male_corpus = corpus.filter_by_gender('male')
    female_corpus = corpus.filter_by_gender('female')
    return compare_word_association_between_corpus_analysis_dunning(word=['money', 'dollars',
                                                                       'pounds',
                                                                   'euros', 'dollar', 'pound',
                                                                   'euro', 'wealth', 'income'],
                                                             corpus1=female_corpus, corpus2=male_corpus,
                                                                    to_pickle=to_pickle)


# America as written by Male Authors versus Female Authors
####################################################################

def america_author_gender_differences(corpus, to_pickle=False):
    """
    Compares how American male authors versus female authors refer to America by looking at the words
    that follow 'America'

    :param corpus: Corpus
    :param to_pickle
    :return:
    """
    if 'author_gender' not in corpus.metadata_fields:
        raise MissingMetadataError(['author_gender'])

    male_corpus = corpus.filter_by_gender('male')
    female_corpus = corpus.filter_by_gender('female')
    return compare_word_association_between_corpus_analysis_dunning(word='America',
                                                                    corpus1=female_corpus,
                                                                    corpus2=male_corpus,
                                                                    to_pickle=to_pickle)


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
    displays bar plot of relative frequency in corpus 2 of all words in results
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
