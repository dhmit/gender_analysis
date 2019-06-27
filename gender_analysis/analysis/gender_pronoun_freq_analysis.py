import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gender_analysis import common
from gender_analysis.analysis import statistical
from gender_analysis.analysis.analysis import get_comparative_word_freq


palette = "colorblind"
style_name = "white"
style_list = {'axes.edgecolor': '.6', 'grid.color': '.9', 'axes.grid': 'True',
              'font.family': 'serif'}
sns.set_color_codes(palette)
sns.set_style(style_name, style_list)


def books_pronoun_freq(corp, to_pickle=False):
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
    {<Document (aanrud_longfrock)>: 0.7623169107856191, <Document (abbott_flatlandromance)>: 0.14321608040201003, <Document (abbott_indiscreetletter)>: 0.4166666666666667, <Document (adams_fighting)>: 0.1898395721925134, <Document (alcott_josboys)>: 0.42152086422368146, <Document (alcott_littlemen)>: 0.3111248200699157, <Document (alcott_littlewomen)>: 0.6196978175713487, <Document (alden_chautauqua)>: 0.7518623169791935, <Document (austen_emma)>: 0.5662100456621004, <Document (austen_persuasion)>: 0.5305111461382571}

    """
    try:
        # relative_freq_male = common.load_pickle(f'{corp.corpus_name}_pronoun_freq_male')
        relative_freq_female = common.load_pickle(f'{corp.name}_pronoun_freq_female')
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

    if to_pickle:
        common.store_pickle(relative_freq_male, f'{corp.corpus_name}_pronoun_freq_male')
        common.store_pickle(relative_freq_female, f'{corp.corpus_name}_pronoun_freq_female')

    return relative_freq_female


def subject_vs_object_pronoun_freqs(corp, to_pickle=False):
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
    ({<Document (aanrud_longfrock)>: 0.793233082706767, <Document (abbott_flatlandromance)>: 0.6741573033707865, <Document (abbott_indiscreetletter)>: 0.7906976744186047, <Document (adams_fighting)>: 0.7184527584020292, <Document (alcott_josboys)>: 0.6330049261083744, <Document (alcott_littlemen)>: 0.6451612903225807, <Document (alcott_littlewomen)>: 0.6577563540753725, <Document (alden_chautauqua)>: 0.7577030812324931, <Document (austen_emma)>: 0.7086120401337792, <Document (austen_persuasion)>: 0.6739130434782609}, {<Document (aanrud_longfrock)>: 0.5376532399299474, <Document (abbott_flatlandromance)>: 0.17543859649122806, <Document (abbott_indiscreetletter)>: 0.4424242424242424, <Document (adams_fighting)>: 0.43485915492957744, <Document (alcott_josboys)>: 0.3862487360970678, <Document (alcott_littlemen)>: 0.4343501326259947, <Document (alcott_littlewomen)>: 0.4124569980083288, <Document (alden_chautauqua)>: 0.5461432506887053, <Document (austen_emma)>: 0.4836730221345606, <Document (austen_persuasion)>: 0.4872013651877133})
    """

    try:
        relative_freq_male_sub_v_ob = common.load_pickle(
            f'{corp.name}_sub_v_ob_pronoun_freq_male')
        relative_freq_female_sub_v_ob = common.load_pickle(
            f'{corp.name}_sub_v_ob_pronoun_freq_female')
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

    if to_pickle:
        common.store_pickle(relative_freq_male_subject,
                            f'{corp.corpus_name}_sub_v_ob_pronoun_freq_male')
        common.store_pickle(relative_freq_female_subject,
                            f'{corp.corpus_name}_sub_v_ob_pronoun_freq_female')

    result_tuple = (relative_freq_male_subject, relative_freq_female_subject)

    return result_tuple


def subject_pronouns_gender_comparison(corp, subject_gender, to_pickle=False):
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
    {<Document (aanrud_longfrock)>: 0.2557575757575758, <Document (abbott_flatlandromance)>: 0.923076923076923, <Document (abbott_indiscreetletter)>: 0.582857142857143, <Document (adams_fighting)>: 0.8210144927536231, <Document (alcott_josboys)>: 0.5736607142857142, <Document (alcott_littlemen)>: 0.6812652068126521, <Document (alcott_littlewomen)>: 0.39719502513892563, <Document (alden_chautauqua)>: 0.2543488481429243, <Document (austen_emma)>: 0.4343926191696566, <Document (austen_persuasion)>: 0.45696623870660963}
    >>> subject_pronouns_gender_comparison(Corpus(filepath, csv_path=csvpath), 'female')
    {<Document (aanrud_longfrock)>: 0.7442424242424243, <Document (abbott_flatlandromance)>: 0.07692307692307691, <Document (abbott_indiscreetletter)>: 0.4171428571428572, <Document (adams_fighting)>: 0.17898550724637682, <Document (alcott_josboys)>: 0.4263392857142857, <Document (alcott_littlemen)>: 0.31873479318734793, <Document (alcott_littlewomen)>: 0.6028049748610743, <Document (alden_chautauqua)>: 0.7456511518570758, <Document (austen_emma)>: 0.5656073808303435, <Document (austen_persuasion)>: 0.5430337612933904}
    """

    if not(subject_gender == 'male' or subject_gender == 'female'):
        raise ValueError('subject_gender must be \'male\' or \'female\'')

    try:
        relative_freq_male_subject = common.load_pickle(
            f'{corp.name}_subject_pronoun_freq_male')
        relative_freq_female_subject = common.load_pickle(
            f'{corp.name}_subject_pronoun_freq_female')
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

    if to_pickle:
        common.store_pickle(relative_freq_female_sub,
                            f'{corp.corpus_name}_subject_pronoun_freq_female')
        common.store_pickle(relative_freq_male_sub, f'{corp.corpus_name}_subject_pronoun_freq_male')

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
    >>> novel_metadata = {'author': 'Brontë, Anne', 'title': 'The Tenant of Wildfell Hall',
    ...                   'corpus_name': 'sample_novels', 'date': '1848', 'author_gender':'female',
    ...                   'filename': 'bronte_wildfell.txt', 'filepath': 'testing/corpora/sample_novels/texts/bronte_wildfell.txt'}
    >>> bronte = document.Document(novel_metadata)
    >>> novel_metadata = {'author': 'Adams, William Taylor', 'title': 'Fighting for the Right',
    ...                   'corpus_name': 'sample_novels', 'date': '1892', 'author_gender':'male',
    ...                   'filename': 'adams_fighting.txt', 'filepath': 'testing/corpora/sample_novels/texts/adams_fighting.txt'}
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
    >>> from gender_analysis.analysis.gender_pronoun_freq_analysis import freq_by_date
    >>> novel_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion',
    ...                   'corpus_name': 'sample_novels', 'date': '1818',
    ...                   'filename': 'austen_persuasion.txt', 'filepath': 'testing/corpora/sample_novels/texts/austen_persuasion.txt'}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ...                   'corpus_name': 'sample_novels', 'date': '1900',
    ...                   'filename': 'hawthorne_scarlet.txt', 'filepath': 'testing/corpora/sample_novels/texts/hawthorne_scarlet.txt'}
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
    >>> from gender_analysis.analysis.gender_pronoun_freq_analysis import freq_by_location
    >>> novel_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion',
    ...                   'corpus_name': 'sample_novels', 'date': '1818',
    ...                   'country_publication': 'United Kingdom', 'filename':  'austen_persuasion.txt',
    ...                   'filepath': 'testing/corpora/sample_novels/texts/austen_persuasion.txt'}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata2 = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ...                   'corpus_name': 'sample_novels', 'date': '1900',
    ...                   'country_publication': 'United States', 'filename':'hawthorne_scarlet.txt',
    ...                   'filepath': 'testing/corpora/sample_novels/texts/hawthorne_scarlet.txt'}
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
    >>> novel_metadata = {'author': 'Austen, Jane', 'title': 'Persuasion',
    ...                   'corpus_name': 'sample_novels', 'date': '1818',
    ...                   'filename': 'austen_persuasion.txt', 'filepath': 'testing/corpora/sample_novels/texts/austen_persuasion.txt'}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ...                   'corpus_name': 'sample_novels', 'date': '1900',
    ...                   'filename': 'hawthorne_scarlet.txt', 'filepath': 'testing/corpora/sample_novels/texts/hawthorne_scarlet.txt'}
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
    '0.47129'
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

    # subject_pronouns_gender_comparison(Corpus('gutenberg'),'female')
