from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import nltk

from gender_analysis import common
from gender_analysis.analysis import statistical


def get_count_words(document, words):
    """
    Takes in a Document object and a list of words to be counted.
    Returns a dictionary where the keys are the elements of 'words' list
    and the values are the numbers of occurrences of the elements in the document.

    Not case-sensitive.

    :param document: Document object
    :param words: a list of words to be counted in text
    :return: a dictionary where the key is the word and the value is the count

    >>> from gender_analysis import document
    >>> from gender_analysis import common
    >>> from pathlib import Path
    >>> document_metadata = {'filename': 'test_text_2.txt',
    ...                      'filepath': Path(common.TEST_DATA_PATH, 'document_test_files', 'test_text_2.txt')}
    >>> doc = document.Document(document_metadata)
    >>> get_count_words(doc, ['sad', 'and'])
    {'sad': 4, 'and': 4}

    """
    dic_word_counts = {}
    for word in words:
        dic_word_counts[word] = document.get_count_of_word(word)
    return dic_word_counts


def get_comparative_word_freq(freqs):
    """
    Returns a dictionary of the frequency of identifier(s) counted relative to each other.
    If frequency passed in is zero, returns zero

    :param freqs: dictionary in the form {'identifier(s)':overall_frequency}
    :return: dictionary in the form {'identifier(s)':relative_frequency}

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'filename': 'hawthorne_scarlet.txt',
    ...                      'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'hawthorne_scarlet.txt')}
    >>> scarlet = document.Document(document_metadata)
    >>> d = {'he':scarlet.get_word_freq('he'), 'she':scarlet.get_word_freq('she')}
    >>> d
    {'he': 0.0073307821095431715, 'she': 0.005895718727577134}
    >>> x = get_comparative_word_freq(d)
    >>> x
    {'he': 0.554249547920434, 'she': 0.445750452079566}
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

    :param freqs: Counter object of words mapped to their word count
    :return: dictionary with key as part of speech, value as Counter object of words
        (of that part of speech) mapped to their word count

    >>> get_counts_by_pos(Counter({'baked':1,'chair':3,'swimming':4}))
    {'VBN': Counter({'baked': 1}), 'NN': Counter({'chair': 3}), 'VBG': Counter({'swimming': 4})}
    >>> get_counts_by_pos(Counter({'is':10,'usually':7,'quietly':42}))
    {'RB': Counter({'quietly': 42, 'usually': 7})}

    """
    common.download_nltk_package_if_not_present('corpora/stopwords')

    sorted_words = {}
    # for each word in the counter
    for word in freqs.keys():
        # filter out if in nltk's list of stop words, e.g. 'is', 'the'
        stop_words = set(nltk.corpus.stopwords.words('english'))
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

    :param d: dictionary in the format {"Author/Document": {'Gender.label : frequency, ...'} ...}
    :param title: title of graph (right now, this is usually just a number)
    :return: None

    """

    book_labels = []
    values = {}
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']

    result_key_list = list(d.keys())
    first_key = result_key_list[0]
    genders = list(d[first_key].keys())

    # I can't work out how to do this better, but I don't love that we're relying on the order
    # of unlinked objects here (book_labels and the lists in values[gender]) to line up right.
    # It should be fine, but there's probably a good way to refactor this.

    for gender in genders:
        values[gender] = []

    for book in d:
        book_labels.append(book)
        for gender in genders:
            if values[gender] == []:
                values[gender] = [d[book][gender]]
            else:
                values[gender].append(d[book][gender])

    fig, ax = plt.subplots()
    plt.ylim(0, 1)

    index = np.arange(len(d.keys()))
    bar_width = .70/len(genders)
    opacity = .5

    lines = {}
    for x in range(0, len(genders)):
        current_gender = genders[x]
        lines[current_gender] = ax.bar(index + x*bar_width, values[genders[x]], bar_width, alpha=opacity, color=colors[x], label=current_gender)

    ax.set_xlabel('Documents')
    ax.set_ylabel('Frequency')
    ax.set_title('Gendered Identifiers by Author')
    ax.set_xticks(index + bar_width*(len(genders)-1) / len(genders))
    plt.xticks(fontsize=8, rotation=90)
    ax.set_xticklabels(book_labels)
    ax.legend()

    fig.tight_layout()
    # TODO: Handle 'this file already exists' gracefully - right now, we don't have permissions to\
    #  overwrite and so we crash. Easiest solution is to just slap a timestamp on here.

    filepng = "visualizations/gender_freq_" + title + ".png"
    filepdf = "visualizations/gender_freq_" + title + ".pdf"
    plt.savefig(filepng, bbox_inches='tight')
    plt.savefig(filepdf, bbox_inches='tight')


def run_gender_freq(corpus, genders):
    """
    Runs a program that uses the gender frequency analysis on all documents existing in a given
    corpus, and outputs the data as graphs

    :param corpus: Corpus
    :param genders: A list of Gender objects to look for
    :return: None

    """
    documents = corpus.documents
    c = len(documents)
    loops = c//10 if c % 10 == 0 else c//10 + 1

    num = 0

    count = {}
    while num < loops:
        dictionary = {}
        for doc in documents[num * 10: min(c, num * 10 + 9)]:
            for gender in genders:
                count[gender] = 0
                for identifier in gender.identifiers:
                    count[gender] += doc.get_word_freq(identifier)

            comp_id_freq = get_comparative_word_freq(count)

            title = hasattr(doc, 'title')
            author = hasattr(doc, 'author')

            if title and author:
                doc_label = doc.title[0:20] + "\n" + doc.author
            elif title:
                doc_label = doc.title[0:20]
            else:
                doc_label = doc.filename[0:20]

            dictionary[doc_label] = comp_id_freq

        display_gender_freq(dictionary, str(num))

        num += 1


def document_pronoun_freq(corp, genders, pickle_filepath=None):
    """
    Counts gendered identifiers for every document in a given corpus,
    and finds their relative frequencies

    Returns a dictionary mapping each Document in the Corpus to the relative frequency of
    gendered identifiers in that Document

    :param corp: Corpus object
    :param genders: A list of Gender objects
    :return: dictionary with data organized by Document

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.analysis.gender_frequency import document_pronoun_freq
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH as path, SMALL_TEST_CORPUS_CSV as path_to_csv
    >>> c = Corpus(path, csv_path=path_to_csv, ignore_warnings = True)
    >>> pronoun_freq_dict = document_pronoun_freq(c)
    >>> flatland = c.get_document('title', 'Flatland')
    >>> result = pronoun_freq_dict[flatland]
    >>> pronoun_freq_dict[flatland]
    {'Male': 0.8493150684931506, 'Female': 0.15068493150684928}

    """

    relative_freqs = {}
    frequencies = {}

    for doc in corp.documents:
        for gender in genders:
            frequencies[gender.label] = 0
            for identifier in gender.identifiers:
                frequencies[gender.label] += doc.get_word_freq(identifier)

        comp_freq_dict = get_comparative_word_freq(frequencies)

        relative_freqs[doc] = comp_freq_dict

    if pickle_filepath:
        common.store_pickle(relative_freqs, pickle_filepath)

    return relative_freqs


def subject_vs_object_pronoun_freqs(corp, pickle_filepath_male=None, pickle_filepath_female=None):
    """
    Takes in a Corpus of novels
    Returns a tuple of two dictionaries, one male and female
    Each dictionary maps each Document in the corpus to the proportion of the pronouns
    of the specified gender in that novel that are subject pronouns

    :param corp: Corpus object
    :param pickle_filepath_female: Location to store results for male results; will not write a file if None
    :param pickle_filepath_male: Location to store results for female results; will not write a file if None
    :return: tuple of two dictionaries (male, female)

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH as path, SMALL_TEST_CORPUS_CSV as path_to_csv
    >>> subject_vs_object_pronoun_freqs(Corpus(path, csv_path=path_to_csv, ignore_warnings = True))
    ({<Document (aanrud_longfrock)>: 0.793233082706767, <Document (abbott_flatlandromance)>: 0.6740331491712708, <Document (abbott_indiscreetletter)>: 0.7906976744186047, <Document (adams_fighting)>: 0.7184527584020292, <Document (alcott_josboys)>: 0.6334563345633457, <Document (alcott_littlemen)>: 0.6441057821604661, <Document (alcott_littlewomen)>: 0.6577563540753725, <Document (alden_chautauqua)>: 0.7577030812324931, <Document (austen_emma)>: 0.7035685320356854, <Document (austen_persuasion)>: 0.6739130434782609}, {<Document (aanrud_longfrock)>: 0.5376532399299474, <Document (abbott_flatlandromance)>: 0.20338983050847456, <Document (abbott_indiscreetletter)>: 0.4424242424242424, <Document (adams_fighting)>: 0.43485915492957744, <Document (alcott_josboys)>: 0.3860535624052551, <Document (alcott_littlemen)>: 0.43594771241830066, <Document (alcott_littlewomen)>: 0.4124569980083288, <Document (alden_chautauqua)>: 0.5461432506887053, <Document (austen_emma)>: 0.48304169367033917, <Document (austen_persuasion)>: 0.4872013651877133})

    """

    relative_freq_male_subject = {}
    relative_freq_female_subject = {}
    relative_freq_male_object = {}
    relative_freq_female_object = {}

    for book in corp.documents:
        # pronouns are hard-coded because these are the only ones guaranteed as subjects and objects
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

    if pickle_filepath_male and pickle_filepath_female:
        common.store_pickle(relative_freq_male_subject, pickle_filepath_male)
        common.store_pickle(relative_freq_female_subject, pickle_filepath_female)

    return relative_freq_male_subject, relative_freq_female_subject


def subject_pronouns_gender_comparison(corp, subject_gender, pickle_filepath_male=None, pickle_filepath_female=None):
    """
    Takes in a Corpus of novels and a gender.
    The gender determines whether the male frequency or female frequency will be returned.

    Returns a dictionary of each novel in the Corpus mapped to the portion of the subject
    pronouns in the book that are of the specified gender.

    :param corp: Corpus object
    :param subject_gender: string 'male' or string 'female'
    :param pickle_filepath_male: Location to store results for male results; will not write a file if None
    :param pickle_filepath_female: Location to store results for female results; will not write a file if None
    :return: dictionary

    >>> from gender_analysis.corpus import Corpus
    >>> from gender_analysis.testing.common import TEST_CORPUS_PATH as path, SMALL_TEST_CORPUS_CSV as path_to_csv
    >>> subject_pronouns_gender_comparison(Corpus(path, csv_path=path_to_csv, ignore_warnings = True), 'male')
    {<Document (aanrud_longfrock)>: 0.2557575757575758, <Document (abbott_flatlandromance)>: 0.9104477611940299, <Document (abbott_indiscreetletter)>: 0.582857142857143, <Document (adams_fighting)>: 0.8210144927536231, <Document (alcott_josboys)>: 0.5741360089186176, <Document (alcott_littlemen)>: 0.6829847908745247, <Document (alcott_littlewomen)>: 0.39719502513892563, <Document (alden_chautauqua)>: 0.2543488481429243, <Document (austen_emma)>: 0.43691765298413493, <Document (austen_persuasion)>: 0.45696623870660963}
    >>> subject_pronouns_gender_comparison(Corpus(path, csv_path=path_to_csv, ignore_warnings = True), 'female')
    {<Document (aanrud_longfrock)>: 0.7442424242424243, <Document (abbott_flatlandromance)>: 0.08955223880597016, <Document (abbott_indiscreetletter)>: 0.4171428571428572, <Document (adams_fighting)>: 0.17898550724637682, <Document (alcott_josboys)>: 0.42586399108138234, <Document (alcott_littlemen)>: 0.31701520912547526, <Document (alcott_littlewomen)>: 0.6028049748610743, <Document (alden_chautauqua)>: 0.7456511518570758, <Document (austen_emma)>: 0.563082347015865, <Document (austen_persuasion)>: 0.5430337612933904}

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

def freq_by_author_gender(d):
    """
    Takes in a dictionary of novel objects mapped to relative frequencies (from **document_pronoun_freq**,
    **subject_vs_object_freqs**, or **subject_pronouns_gender_comparison**).
    Returns a dictionary with frequencies binned by author gender into lists.

    list names key:
    male_author - male authors
    female_author- female authors

    :param d: dictionary
    :return: dictionary

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> novel_metadata = {'author': 'Brontë, Anne', 'title': 'The Tenant of Wildfell Hall', 'date': '1848', 'author_gender':'female',
    ...                   'filename': 'bronte_wildfell.txt', 'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'bronte_wildfell.txt')}
    >>> bronte = document.Document(novel_metadata)
    >>> novel_metadata = {'author': 'Adams, William Taylor', 'title': 'Fighting for the Right', 'date': '1892', 'author_gender':'male',
    ...                   'filename': 'adams_fighting.txt', 'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'adams_fighting.txt')}
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
    Takes in a dictionary of novel objects mapped to relative frequencies, and
    returns a dictionary with frequencies binned by decades into lists
    List name is mapped to the list of frequencies

    list names key:
    date_to_1810 - publication dates before and not including 1810
    date_x_to_y (by decade) - publication dates from x to y
    Example: date_1810_to_1819 - publication dates from 1810 to 1819
    date_1900_on - publication dates in 1900 and onward

    :param d: dictionary
    :param time_frame: tuple (int start year, int end year) for the range of dates to return frequencies
    :param bin_size: int for the number of years represented in each list of frequencies
    :return: dictionary {bin_start_year:[frequencies for documents in this bin of years]

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> from gender_analysis.analysis.gender_frequency import freq_by_date
    >>> novel_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
    ...                   'filename': 'austen_persuasion.txt', 'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'austen_persuasion.txt')}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1900',
    ...                   'filename': 'hawthorne_scarlet.txt', 'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'hawthorne_scarlet.txt')}
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
    Takes in a dictionary of novel objects mapped to relative frequencies.
    Returns a dictionary with frequencies binned by publication location into lists

    list names key:

    - *location_UK* - published in the United Kingdom
    - *location_US* - published in the US
    - *location_other* - published somewhere other than the US and England

    :param d: dictionary
    :return: dictionary

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> from gender_analysis.analysis.gender_frequency import freq_by_location
    >>> novel_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
    ...                   'country_publication': 'United Kingdom', 'filename':  'austen_persuasion.txt',
    ...                   'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'austen_persuasion.txt')}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata2 = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1900',
    ...                   'country_publication': 'United States', 'filename':'hawthorne_scarlet.txt',
    ...                   'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'hawthorne_scarlet.txt')}
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
    Takes in a dictionary of documents mapped to pronoun frequencies and returns a dictionary of
    years mapped to lists of pronoun frequencies

    :param frequency_dict: dictionary of documents mapped to pronoun frequencies
    :return: dictionary of years mapped to lists of pronoun frequencies

    >>> from gender_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> novel_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion', 'date': '1818',
    ...                   'filename': 'austen_persuasion.txt', 'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'austen_persuasion.txt')}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter', 'date': '1900',
    ...                   'filename': 'hawthorne_scarlet.txt', 'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts', 'hawthorne_scarlet.txt')}
    >>> scarlet = document.Document(novel_metadata)
    >>> d = {scarlet:0.5, austen:0.3}
    >>> sorted_years = sort_every_year(d)
    >>> print(sorted_years)
    {1900: [0.5], 1818: [0.3]}

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
    Takes in a frequency dictionary (from **freq_by_author_gender**, **freq_by_date**,
     or **freq_by_location**) and exports its values as a bar-and-whisker graph

    :param freq_dict: dictionary of frequencies
    :param my_pal: palette to be used
    :param title: title of exported graph
    :param x: name of x-vars
    :return: None

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
    common.load_graph_settings()
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
    Creates a bar graph given male/female subject/object frequencies. Meant to be run with data
    sorted by 'freq_by_author_gender', 'freq_by_date', or 'freq_by_location'

    :param she_freq_dict:
    :param he_freq_dict:
    :param title: name of the exported file
    :param x: value of x axis
    :return: None

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

def stat_analysis(corpus):
    tot_female_dict = document_pronoun_freq(corpus)
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
