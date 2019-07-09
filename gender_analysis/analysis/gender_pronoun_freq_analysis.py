import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from collections import Counter

from gender_analysis import common
from gender_analysis.analysis import statistical

nltk.download('stopwords', quiet=True)

palette = "colorblind"
style_name = "white"
style_list = {'axes.edgecolor': '.6', 'grid.color': '.9', 'axes.grid': 'True',
              'font.family': 'serif'}
sns.set_color_codes(palette)
sns.set_style(style_name, style_list)


def get_count_words(document, words):
    """
    Takes in document, a Document object, and words, a list of words to be counted.
    Returns a dictionary where the keys are the elements of 'words' list
    and the values are the numbers of occurrences of the elements in the document.
    N.B.: Not case-sensitive.
    >>> from gender_analysis import document
    >>> from gender_analysis import common
    >>> from pathlib import Path
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1850',
    ...                   'filename': 'test_text_2.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'document_test_files', 'test_text_2.txt')}
    >>> scarlett = document.Document(document_metadata)
    >>> get_count_words(scarlett, ["sad", "and"])
    {'sad': 4, 'and': 4}

    :param: words: a list of words to be counted in text
    :return: a dictionary where the key is the word and the value is the count
    """
    dic_word_counts = {}
    for word in words:
        dic_word_counts[word] = document.get_count_of_word(word)
    return dic_word_counts


def get_comparative_word_freq(freqs):
    """
    Returns a dictionary of the frequency of words counted relative to each other.
    If frequency passed in is zero, returns zero

    :param freqs: dictionary in the form {'word':overall_frequency}
    :return: dictionary in the form {'word':relative_frequency}

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1900',
    ...                   'filename': 'hawthorne_scarlet.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'hawthorne_scarlet.txt')}
    >>> scarlet = document.Document(document_metadata)
    >>> d = {'he':scarlet.get_word_freq('he'), 'she':scarlet.get_word_freq('she')}
    >>> d
    {'he': 0.007099649057997783, 'she': 0.005702807536017732}
    >>> x = get_comparative_word_freq(d)
    >>> x
    {'he': 0.5545536519386836, 'she': 0.44544634806131655}
    >>> d2 = {'he': 0, 'she': 0}
    >>> d2
    {'he': 0, 'she': 0}
    """

    total_freq = sum(freqs.values())
    comp_freqs = {}

    for k, v in freqs.items():
        try:
            freq = v / total_freq
        except ZeroDivisionError:
            freq = 0
        comp_freqs[k] = freq

    return comp_freqs


def get_counts_by_pos(freqs):
    """
    This functions returns a dictionary where each key is a part of speech tag (e.g. 'NN' for nouns)
    and the value is a counter object of words of that part of speech and their frequencies.
    It also filters out words like "is", "the". We used `nltk`'s stop words function for filtering.

    >>> get_counts_by_pos(Counter({'baked':1,'chair':3,'swimming':4}))
    {'VBN': Counter({'baked': 1}), 'NN': Counter({'chair': 3}), 'VBG': Counter({'swimming': 4})}
    >>> get_counts_by_pos(Counter({'is':10,'usually':7,'quietly':42}))
    {'RB': Counter({'quietly': 42, 'usually': 7})}

    :param freqs: Counter object of words mapped to their word count
    :return: dictionary with key as part of speech, value as Counter object of words (of that
    part of speech) mapped to their word count
    """

    sorted_words = {}
    # for each word in the counter
    for word in freqs.keys():
        # filter out if in nltk's list of stop words, e.g. is, the
        stop_words = set(stopwords.words('english'))
        if word not in stop_words:
            # get its part of speech tag from nltk's pos_tag function
            tag = nltk.pos_tag([word])[0][1]
            # add that word to the counter object in the relevant dict entry
            if tag not in sorted_words.keys():
                sorted_words[tag] = Counter({word:freqs[word]})
            else:
                sorted_words[tag].update({word: freqs[word]})
    return sorted_words


def display_gender_freq(d, title):
    """
    Takes in a dictionary sorted by author and gender frequencies, and a title.
    Outputs the resulting graph to 'visualizations/title.pdf' AND 'visualizations/title.png'

    Will scale to allow inputs of larger dictionaries with non-binary values

    :param d: dictionary in the format {"Author/Document": [he_freq, she_freq]}
    :param title: title of graph
    :return:
    """
    he_val = []
    she_val = []
    authors = []

    for entry in d:
        authors.append(entry)
        he_val.append(d[entry][0])
        she_val.append(d[entry][1])

    fig, ax = plt.subplots()
    plt.ylim(0, 1)

    index = np.arange(len(d.keys()))
    bar_width = 0.35
    opacity = 0.4

    he_val = tuple(he_val)
    she_val = tuple(she_val)
    authors = tuple(authors)

    rects1 = ax.bar(index, he_val, bar_width, alpha=opacity, color='b', label='He')
    rects2 = ax.bar(index + bar_width, she_val, bar_width, alpha=opacity, color='r', label='She')

    ax.set_xlabel('Authors')
    ax.set_ylabel('Frequency')
    ax.set_title('Gendered Pronouns by Author')
    ax.set_xticks(index + bar_width / 2)
    plt.xticks(fontsize=8, rotation=90)
    ax.set_xticklabels(authors)
    ax.legend()

    fig.tight_layout()
    filepng = "visualizations/he_she_freq" + title + ".png"
    filepdf = "visualizations/he_she_freq" + title + ".pdf"
    plt.savefig(filepng, bbox_inches='tight')
    plt.savefig(filepdf, bbox_inches='tight')


def run_gender_freq(corpus):
    """
    Runs a program that uses the gender frequency analysis on all documents existing in a given
    corpus, and outputs the data as graphs
    :param corpus: Corpus
    :return:
    """
    documents = corpus.documents
    c = len(documents)
    loops = c//10 if c % 10 == 0 else c//10 + 1

    num = 0

    while num < loops:
        dictionary = {}
        for doc in documents[num * 10: min(c, num * 10 + 9)]:
            d = {'he': doc.get_word_freq('he'), 'she': doc.get_word_freq('she')}
            d = get_comparative_word_freq(d)
            lst = [d["he"], d["she"]]
            title = hasattr(doc, 'title')
            author = hasattr(doc, 'author')
            if title and author:
                doc_label = doc.title[0:20] + "\n" + doc.author
            elif title:
                doc_label = doc.title[0:20]
            else:
                doc_label = doc.filename[0:20]
            dictionary[doc_label] = lst
        display_gender_freq(dictionary, str(num))
        num += 1


def books_pronoun_freq(corp, pickle_filepath=None):
    """
    Counts male and female pronouns for every book and finds their relative frequencies per book
    Outputs dictionary mapping novel object to the relative frequency
        of female pronouns in that book

    :param: corp: Corpus object
    :return: dictionary with data organized by groups

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.analysis.gender_pronoun_freq_analysis import books_pronoun_freq
    >>> from gender_analysis.common import BASE_PATH
    >>> filepath = BASE_PATH / 'testing' / 'corpora' / 'test_corpus'
    >>> csvpath = BASE_PATH / 'testing' / 'corpora' / 'test_corpus' / 'test_corpus.csv'
    >>> books_pronoun_freq(Corpus(filepath, csv_path=csvpath))
    {<Document (aanrud_longfrock)>: 0.7614617940199335, <Document (abbott_flatlandromance)>: 0.14463840399002492, <Document (abbott_indiscreetletter)>: 0.4160401002506266, <Document (adams_fighting)>: 0.18998330550918197, <Document (alcott_josboys)>: 0.4214648602878916, <Document (alcott_littlemen)>: 0.31113851212494864, <Document (alcott_littlewomen)>: 0.6196017006041621, <Document (alden_chautauqua)>: 0.7515400410677618, <Document (austen_emma)>: 0.566123858869973, <Document (austen_persuasion)>: 0.5303780378037803}
    """
    try:
        relative_freq_female = common.load_pickle(pickle_filepath)
        return relative_freq_female
    except IOError:
        pass

    relative_freq_male = {}
    relative_freq_female = {}

    for book in corp.documents:
        he = book.get_word_freq('he')
        him = book.get_word_freq('him')
        his = book.get_word_freq('his')
        male = he + him + his

        she = book.get_word_freq('she')
        her = book.get_word_freq('her')
        hers = book.get_word_freq('hers')
        female = she + her + hers

        temp_dict = {'male': male, 'female': female}
        temp_dict = get_comparative_word_freq(temp_dict)

        relative_freq_male[book] = temp_dict['male']
        relative_freq_female[book] = temp_dict['female']

    book.text = ''
    book._word_counts_counter = None

    if pickle_filepath:
        common.store_pickle(relative_freq_male, pickle_filepath)
        common.store_pickle(relative_freq_female, pickle_filepath)

    return relative_freq_female


def subject_vs_object_pronoun_freqs(corp, pickle_filepath_male=None, pickle_filepath_female=None):
    """
    Takes in a Corpus of novels
    Returns a tuple of two dictionaries, one male and female
    Each dictionary maps each Document in the corpus to the proportion of the pronouns
        of the specified gender in that novel that are subject pronouns

    :param corp: Corpus
    :param to_pickle
    :return: tuple of two dictionaries (male, female)

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.common import BASE_PATH
    >>> filepath = BASE_PATH / 'testing' / 'corpora' / 'test_corpus'
    >>> csvpath = BASE_PATH / 'testing' / 'corpora' / 'test_corpus' / 'test_corpus.csv'
    >>> subject_vs_object_pronoun_freqs(Corpus(filepath, csv_path=csvpath))
    ({<Document (aanrud_longfrock)>: 0.7947761194029851, <Document (abbott_flatlandromance)>: 0.6777777777777777, <Document (abbott_indiscreetletter)>: 0.7938931297709924, <Document (adams_fighting)>: 0.7188093730208993, <Document (alcott_josboys)>: 0.6334563345633456, <Document (alcott_littlemen)>: 0.6454880294659301, <Document (alcott_littlewomen)>: 0.6580560420315237, <Document (alden_chautauqua)>: 0.7583798882681564, <Document (austen_emma)>: 0.7088554720133667, <Document (austen_persuasion)>: 0.6743697478991596}, {<Document (aanrud_longfrock)>: 0.5380577427821522, <Document (abbott_flatlandromance)>: 0.18965517241379312, <Document (abbott_indiscreetletter)>: 0.4457831325301205, <Document (adams_fighting)>: 0.4358523725834798, <Document (alcott_josboys)>: 0.38655886811520973, <Document (alcott_littlemen)>: 0.43472498343273697, <Document (alcott_littlewomen)>: 0.41256335988414194, <Document (alden_chautauqua)>: 0.5462994836488813, <Document (austen_emma)>: 0.48378615249780893, <Document (austen_persuasion)>: 0.48742004264392325})
    """

    try:
        relative_freq_male_sub_v_ob = common.load_pickle(pickle_filepath_male)
        relative_freq_female_sub_v_ob = common.load_pickle(pickle_filepath_female)
        return relative_freq_male_sub_v_ob, relative_freq_female_sub_v_ob
    except IOError:
        pass

    relative_freq_male_subject = {}
    relative_freq_female_subject = {}
    relative_freq_male_object = {}
    relative_freq_female_object = {}

    for book in corp.documents:
        he = book.get_word_freq('he')
        him = book.get_word_freq('him')

        she = book.get_word_freq('she')
        her = book.get_word_freq('her')

        temp_dict_male = {'subject': he, 'object': him}
        temp_dict_female = {'subject': she, 'object': her}
        temp_dict_male = get_comparative_word_freq(temp_dict_male)
        temp_dict_female = get_comparative_word_freq(temp_dict_female)

        relative_freq_male_subject[book] = temp_dict_male['subject']
        relative_freq_female_subject[book] = temp_dict_female['subject']
        relative_freq_male_object[book] = temp_dict_male['object']
        relative_freq_female_object[book] = temp_dict_female['object']

    book.text = ''
    book._word_counts_counter = None

    if pickle_filepath_male and pickle_filepath_female:
        common.store_pickle(relative_freq_male_subject,
                            pickle_filepath_male)
        common.store_pickle(relative_freq_female_subject,
                            pickle_filepath_female)

    result_tuple = (relative_freq_male_subject, relative_freq_female_subject)

    return result_tuple


def subject_pronouns_gender_comparison(corp, subject_gender, pickle_filepath_male=None, pickle_filepath_female=None):
    """
    Takes in a Corpus of novels and a gender.
    The gender determines whether the male frequency or female frequency will
        be returned
    Returns a dictionary of each novel in the Corpus mapped to the portion of
        the subject pronouns in the book that are of the specified gender
    :param corp: Corpus
    :param subject_gender: string 'male' or string 'female'
    :return: dictionary

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.common import BASE_PATH
    >>> filepath = BASE_PATH / 'testing' / 'corpora' / 'test_corpus'
    >>> csvpath = BASE_PATH / 'testing' / 'corpora' / 'test_corpus' / 'test_corpus.csv'
    >>> subject_pronouns_gender_comparison(Corpus(filepath, csv_path=csvpath), 'male')
    {<Document (aanrud_longfrock)>: 0.25724637681159424, <Document (abbott_flatlandromance)>: 0.9172932330827067, <Document (abbott_indiscreetletter)>: 0.5842696629213483, <Document (adams_fighting)>: 0.8206796818510484, <Document (alcott_josboys)>: 0.573816155988858, <Document (alcott_littlemen)>: 0.6812439261418853, <Document (alcott_littlewomen)>: 0.3974087784241142, <Document (alden_chautauqua)>: 0.2549295774647887, <Document (austen_emma)>: 0.4345710627400768, <Document (austen_persuasion)>: 0.45726495726495725}
    >>> subject_pronouns_gender_comparison(Corpus(filepath, csv_path=csvpath), 'female')
    {<Document (aanrud_longfrock)>: 0.7427536231884058, <Document (abbott_flatlandromance)>: 0.08270676691729323, <Document (abbott_indiscreetletter)>: 0.4157303370786517, <Document (adams_fighting)>: 0.17932031814895155, <Document (alcott_josboys)>: 0.4261838440111421, <Document (alcott_littlemen)>: 0.31875607385811466, <Document (alcott_littlewomen)>: 0.6025912215758857, <Document (alden_chautauqua)>: 0.7450704225352113, <Document (austen_emma)>: 0.5654289372599232, <Document (austen_persuasion)>: 0.5427350427350427}
    """

    if not(subject_gender == 'male' or subject_gender == 'female'):
        raise ValueError('subject_gender must be \'male\' or \'female\'')

    try:
        relative_freq_male_subject = common.load_pickle(pickle_filepath_male)
        relative_freq_female_subject = common.load_pickle(pickle_filepath_female)
        if subject_gender == 'male':
            return relative_freq_male_subject
        else:
            return relative_freq_female_subject
    except IOError:
        pass

    relative_freq_female_sub = {}
    relative_freq_male_sub = {}

    for book in corp.documents:
        he = book.get_word_freq('he')
        she = book.get_word_freq('she')

        relative_freq_female_sub[book] = she/(he+she)
        relative_freq_male_sub[book] = he/(he+she)

    book.text = ''
    book._word_counts_counter = None

    if pickle_filepath_male and pickle_filepath_female:
        common.store_pickle(relative_freq_female_sub,
                            pickle_filepath_female)
        common.store_pickle(relative_freq_male_sub, pickle_filepath_male)

    if subject_gender == 'male':
        return relative_freq_male_sub
    elif subject_gender == 'female':
        return relative_freq_female_sub
    else:
        raise ValueError('subject_gender must be \'male\' or \'female\'')


def dict_to_list(d):
    """
    Takes in a dictionary and returns a list of the values in the dictionary
    If there are repeats in the values, there will be repeats in the list
    :param d: dictionary
    :return: list of values in the dictionary

    >>> d = {'a': 1, 'b': 'bee', 'c': 65}
    >>> dict_to_list(d)
    [1, 'bee', 65]

    >>> d2 = {}
    >>> dict_to_list(d2)
    []
    """
    L = []
    for key, value in d.items():
        L.append(value)
    return L


def freq_by_author_gender(d):
    """
    Takes in a dictionary of novel objects mapped to relative frequencies
        (output of above function)
    Returns a dictionary with frequencies binned by author gender into lists
        List name is mapped to the list of frequencies

    list names key:
    male_author - male authors
    female_author- female authors

    :param d: dictionary
    :return: dictionary

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> novel_metadata = {'author': 'Brontë, Anne', 'title': 'The Tenant of Wildfell Hall', 'date': '1848', 'author_gender':'female',
    ...                   'filename': 'bronte_wildfell.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'bronte_wildfell.txt')}
    >>> bronte = document.Document(novel_metadata)
    >>> novel_metadata = {'author': 'Adams, William Taylor', 'title': 'Fighting for the Right', 'date': '1892', 'author_gender':'male',
    ...                   'filename': 'adams_fighting.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'adams_fighting.txt')}
    >>> fighting = document.Document(novel_metadata)
    >>> d = {fighting:0.3, bronte:0.6}
    >>> freq_by_author_gender(d)
    {'Male Author': [0.3], 'Female Author': [0.6]}
    """

    male_author = []
    female_author = []
    data = {}

    for k, v in d.items():
        author_gender = getattr(k, 'author_gender', None)
        if author_gender == 'male':
            male_author.append(v)

        elif author_gender == 'female':
            female_author.append(v)

    data['Male Author'] = male_author
    data['Female Author'] = female_author

    return data


def freq_by_date(d, time_frame, bin_size):
    """
    Takes in a dictionary of novel objects mapped to relative frequencies
        (output of above function)
    Returns a dictionary with frequencies binned by decades into lists
        List name is mapped to the list of frequencies

    list names key:
    date_to_1810 - publication dates before and not including 1810
    date_x_to_y (by decade) - publication dates from x to y
        Example: date_1810_to_1819 - publication dates from 1810 to 1819
    date_1900_on - publication dates in 1900 and onward

    :param d: dictionary
    :param time_frame: tuple (int start year, int end year) for the range of dates to return
    frequencies
    :param bin_size: int for the number of years represented in each list of frequencies
    :return: dictionary {bin_start_year:[frequencies for documents in this bin of years]

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> from gender_analysis.analysis.gender_pronoun_freq_analysis import freq_by_date
    >>> novel_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
    ...                   'filename': 'austen_persuasion.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'austen_persuasion.txt')}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1900',
    ...                   'filename': 'hawthorne_scarlet.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'hawthorne_scarlet.txt')}
    >>> scarlet = document.Document(novel_metadata)
    >>> d = {scarlet:0.5, austen:0.3}
    >>> freq_by_date(d, (1770, 1910), 10)
    {1770: [], 1780: [], 1790: [], 1800: [], 1810: [0.3], 1820: [], 1830: [], 1840: [], 1850: [], 1860: [], 1870: [], 1880: [], 1890: [], 1900: [0.5]}
    """

    data = {}
    for bin_start_year in range(time_frame[0], time_frame[1], bin_size):
        data[bin_start_year] = []

    for k,v in d.items():
        date = getattr(k, 'date', None)
        if date is None:
            continue
        bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
        data[bin_year].append(v)

    return data


def freq_by_location(d):
    """
    Takes in a dictionary of novel objects mapped to relative frequencies
        (output of above function)
    Returns a dictionary with frequencies binned by publication location into lists
        List name is mapped to the list of frequencies

    list names key:
    location_UK - published in the United Kingdom
    location_US - published in the US
    location_other - published somewhere other than the US and England

    :param d: dictionary
    :return: dictionary

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> from gender_analysis.analysis.gender_pronoun_freq_analysis import freq_by_location
    >>> novel_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
    ...                   'country_publication': 'United Kingdom', 'filename':  'austen_persuasion.txt',
    ...                   'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'austen_persuasion.txt')}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata2 = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1900',
    ...                   'country_publication': 'United States', 'filename':'hawthorne_scarlet.txt',
    ...                   'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'hawthorne_scarlet.txt')}
    >>> scarlet = document.Document(novel_metadata2)
    >>> d = {scarlet:0.5, austen:0.3}
    >>> freq_by_location(d)
    {'United States': [0.5], 'United Kingdom': [0.3]}
    """
    data = {}

    for k, v in d.items():
        location = getattr(k, 'country_publication', None)
        if location is None:
            continue

        if location not in data:
            data[location] = []
        data[location].append(v)

    return data


def get_mean(data_dict):
    """
    Takes in a dictionary matching some object to lists and returns a dictionary of the
        original keys mapped to the mean of the lists

    :param data_dict: dictionary matching some object to lists
    :return: dictionary with original key mapped to an average of the input list

    >>> d = {}
    >>> d['fives'] = [5,5,5]
    >>> d['halfway'] = [0,1]
    >>> d['nothing'] = [0]
    >>> get_mean(d)
    {'fives': 5.0, 'halfway': 0.5, 'nothing': 0.0}
    """
    mean_dict = {}
    for k, v in data_dict.items():
        try:
            mean_dict[k] = np.mean(v)
        except:
            mean_dict[k + "*"] = 0.5
    return mean_dict


def sort_every_year(frequency_dict):
    """
    Takes in a dictionary of novels mapped to pronoun frequencies and returns a dictionay of
        years mapped to lists of pronoun frequencies

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> novel_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
    ...                   'filename': 'austen_persuasion.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'austen_persuasion.txt')}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1900',
    ...                   'filename': 'hawthorne_scarlet.txt', 'filepath': Path(common.BASE_PATH, 'testing', 'corpora', 'sample_novels', 'texts', 'hawthorne_scarlet.txt')}
    >>> scarlet = document.Document(novel_metadata)
    >>> d = {scarlet:0.5, austen:0.3}
    >>> sorted_years = sort_every_year(d)
    >>> print(sorted_years)
    {1900: [0.5], 1818: [0.3]}


    :param frequency_dict: dictionary of novels mapped to pronoun frequencies
    :return: dictionary of years mapped to lists of pronoun frequencies
    """

    every_year_dict = {}
    for key, value in frequency_dict.items():
        # TODO: check if key (document?) has date attribute
        frequency_list = [frequency_dict[key]]

        if key.date not in every_year_dict.keys():
            every_year_dict[key.date] = frequency_list

        elif key.date in every_year_dict.keys():
            every_year_dict[key.date].append(frequency_dict[key])

    return every_year_dict


def box_gender_pronoun_freq(freq_dict, my_pal, title, x="N/A"):
    """
    Takes in a frequency dictionaries and exports its values as a bar-and-whisker graph
    :param freq_dict: dictionary of frequencies grouped up
    :param my_pal: palette to be used
    :param title: title of exported graph
    :param x: name of x-vars
    :return:
    """

    plt.clf()
    groups = []
    val = []
    for k, v in freq_dict.items():
        temp = [k]*len(v)
        groups.extend(temp)
        val.extend(v)

    df = pd.DataFrame({x: groups, 'Frequency': val})
    df = df[[x, 'Frequency']]
    sns.boxplot(x=df[x], y=df['Frequency'],
                palette=my_pal).set_title("Relative Frequency of Female Pronouns to Total Pronouns")
    plt.xticks(rotation=90)
    # plt.show()

    filepng = "visualizations/" + title + ".png"
    filepdf = "visualizations/" + title + ".pdf"
    plt.savefig(filepng, bbox_inches='tight')
    plt.savefig(filepdf, bbox_inches='tight')


def bar_sub_obj_freq(she_freq_dict, he_freq_dict, title, x="N/A"):
    """
    Creates a bar graph give male/female subject/object frequencies. Meant to be run with data
    sorted by 'freq_by_author_gender', 'freq_by_date', or 'freq_by_location'
    :param she_freq_dict:
    :param he_freq_dict:
    :param title: name of the exported file
    :param x: value of x axis
    :return:
    """

    fig, ax = plt.subplots()
    plt.ylim(0, 1)

    key = []

    for k, v in she_freq_dict.items():
        key.append(k)

    m_freq = dict_to_list(he_freq_dict)
    f_freq = dict_to_list(she_freq_dict)

    index = np.arange(len(she_freq_dict.keys()))
    bar_width = 0.35
    opacity = 0.4

    ax.bar(index, [1]*len(m_freq), bar_width, alpha=opacity, color='c', label="Male Object")
    ax.bar(index, m_freq, bar_width, alpha=opacity, color='b', label='Male Subject')
    ax.bar(index + bar_width, [1]*len(f_freq), bar_width, alpha=opacity, color='#DE8F05',
           label="Female Object")
    ax.bar(index + bar_width, f_freq, bar_width, alpha=opacity, color='r', label='Female Subject')

    ax.set_xlabel(x)
    ax.set_ylabel('Frequency')
    ax.set_title('Relative Frequencies of Subject to Object Pronouns')
    ax.set_xticks(index + bar_width / 2)
    plt.xticks(fontsize=8, rotation=90)
    ax.set_xticklabels(key)
    ax.legend()

    fig.tight_layout()

    filepng = "visualizations/" + title + ".png"
    filepdf = "visualizations/" + title + ".pdf"
    plt.savefig(filepng, bbox_inches='tight')
    plt.savefig(filepdf, bbox_inches='tight')


def overall_mean(d):
    """
    Returns the average of all the values in a dictionary
    :param d: dictionary with numbers as values
    :return: float: average of all the values
    >>> from gender_analysis.analysis.gender_pronoun_freq_analysis import overall_mean, books_pronoun_freq
    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.common import BASE_PATH
    >>> filepath = BASE_PATH / 'testing' / 'corpora' / 'test_corpus'
    >>> csvpath = BASE_PATH / 'testing' / 'corpora' / 'test_corpus' / 'test_corpus.csv'
    >>> c = Corpus(filepath, csv_path=csvpath)
    >>> freq = books_pronoun_freq(c)
    >>> mean = overall_mean(freq)
    >>> str(mean)[:7]
    '0.47123'
    """
    l = dict_to_list(d)
    mean = np.mean(l)
    return mean


def stat_analysis(corpus):
    tot_female_dict = books_pronoun_freq(corpus)
    author_to_freq_dict = freq_by_author_gender(tot_female_dict)

    author_gender_pronoun_analysis = statistical.get_p_and_ttest_value(author_to_freq_dict[
                                                                     'male_author'],
                                                           author_to_freq_dict["female_author"])
    print("values for gender pronoun stats: ", author_gender_pronoun_analysis)

    sub_v_ob_tuple = subject_vs_object_pronoun_freqs(corpus)

    sub_v_ob_male_dict = sub_v_ob_tuple[0]
    sub_v_ob_male_list = dict_to_list(sub_v_ob_male_dict)

    sub_v_ob_female_dict = sub_v_ob_tuple[1]
    sub_v__ob_female_list = dict_to_list(sub_v_ob_female_dict)

    author_gender_sub_v_ob_correlation = statistical.get_p_and_ttest_value(sub_v_ob_male_list,
                                                                           sub_v__ob_female_list)
    print("values for subject vs object pronouns between male and female authors: ",
          author_gender_sub_v_ob_correlation)
