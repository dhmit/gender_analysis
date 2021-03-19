from collections import Counter
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import nltk
from gender_analysis import common


def get_count_words(document, words):
    """
    Takes in a Document object and a list of words to be counted.
    Returns a dictionary where the keys are the elements of 'words' list
    and the values are the numbers of occurrences of the elements in the document.

    Not case-sensitive.

    :param document: Document object
    :param words: a list of words to be counted in text
    :return: a dictionary where the key is the word and the value is the count

    >>> from corpus_analysis import document
    >>> from gender_analysis import common
    >>> from pathlib import Path
    >>> document_metadata = {'filename': 'test_text_2.txt',
    ...                      'filepath': Path(common.TEST_DATA_PATH, 'document_test_files',
    ...                      'test_text_2.txt')}
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

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> document_metadata = {'filename': 'hawthorne_scarlet.txt',
    ...                      'filepath': Path(common.TEST_DATA_PATH, 'sample_novels', 'texts',
    ...                      'hawthorne_scarlet.txt')}
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

    for k, ori_freq in freqs.items():
        try:
            freq = ori_freq / total_freq
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
                sorted_words[tag] = Counter({word: freqs[word]})
            else:
                sorted_words[tag].update({word: freqs[word]})
    return sorted_words


def display_gender_freq(freq_dic, title):
    # pylint: disable=too-many-locals
    """
    Takes in a dictionary sorted by author and gender frequencies, and a title.
    Outputs the resulting graph to 'visualizations/title.pdf' AND 'visualizations/title.png'

    Will scale to allow inputs of larger dictionaries with non-binary values

    :param freq_dic: dictionary in the format {"Author/Document":
                                                   {'Gender.label : frequency, ...'} ...}
    :param title: title of graph (right now, this is usually just a number)
    :return: None

    """

    book_labels = []
    values = {}
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']

    result_key_list = list(freq_dic.keys())
    first_key = result_key_list[0]
    genders = list(freq_dic[first_key].keys())

    # I can't work out how to do this better, but I don't love that we're relying on the order
    # of unlinked objects here (book_labels and the lists in values[gender]) to line up right.
    # It should be fine, but there's probably a good way to refactor this.

    for gender in genders:
        values[gender] = []

    for book in freq_dic:
        book_labels.append(book)
        for gender in genders:
            if not values[gender]:
                values[gender] = [freq_dic[book][gender]]
            else:
                values[gender].append(freq_dic[book][gender])

    fig, axis = plt.subplots()
    plt.ylim(0, 1)

    index = np.arange(len(freq_dic.keys()))
    bar_width = .70 / len(genders)
    opacity = .5
    count = 0
    lines = {}
    for gender in genders:
        current_gender = gender
        lines[current_gender] = axis.bar(index + count * bar_width, values[genders[count]],
                                         bar_width, alpha=opacity, color=colors[count],
                                         label=current_gender)
        count += 1

    axis.set_xlabel('Documents')
    axis.set_ylabel('Frequency')
    axis.set_title('Gendered Identifiers by Author')
    axis.set_xticks(index + bar_width * (len(genders) - 1) / len(genders))
    plt.xticks(fontsize=8, rotation=90)
    axis.set_xticklabels(book_labels)
    axis.legend()

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
    num_documents = len(documents)
    loops = num_documents // 10 if num_documents % 10 == 0 else num_documents // 10 + 1

    num = 0

    count = {}
    while num < loops:
        dictionary = {}
        for doc in documents[num * 10: min(num_documents, num * 10 + 9)]:
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


def corpus_pronoun_freq(corp, genders, pickle_filepath=None):
    """
    Counts gendered identifiers for every document in a given corpus,
    and finds their relative frequencies

    Returns a dictionary mapping each Document in the Corpus to the relative frequency of
    gendered identifiers in that Document

    :param corp: Corpus object
    :param genders: A list of Gender objects
    :param pickle_filepath: Filepath to save the pickled results.
    :return: dictionary with data organized by Document

    >>> from corpus_analysis.corpus import Corpus
    >>> from gender_analysis.analysis.gender_frequency import corpus_pronoun_freq
    >>> from gender_analysis.common import BINARY_GROUP
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH, SMALL_TEST_CORPUS_CSV
    >>> c = Corpus(TEST_CORPUS_PATH, csv_path=SMALL_TEST_CORPUS_CSV, ignore_warnings = True)
    >>> pronoun_freq_dict = corpus_pronoun_freq(c, BINARY_GROUP)
    >>> flatland = c.get_document('title', 'Flatland')
    >>> result = pronoun_freq_dict[flatland]
    >>> pronoun_freq_dict[flatland]
    {'Female': 0.1494252873563218, 'Male': 0.8505747126436781}

    """

    relative_freqs = {}

    for doc in corp.documents:
        comp_freq_dict = doc_pronoun_freq(doc, genders)
        relative_freqs[doc] = comp_freq_dict

    if pickle_filepath:
        common.store_pickle(relative_freqs, pickle_filepath)

    return relative_freqs


def doc_pronoun_freq(document, genders):

    """
    Counts gendered identifiers in a document and finds their relative frequencies

    Returns a dictionary mapping each Document in the Corpus to the relative frequency of
    gendered identifiers in that Document

    :param document: Document object
    :param genders: A list of Gender objects
    :return: dictionary with data organized by gender
    """

    frequencies = {}

    for gender in genders:
        frequencies[gender.label] = 0
        for identifier in gender.identifiers:
            frequencies[gender.label] += document.get_word_freq(identifier)

    comp_freq_dict = get_comparative_word_freq(frequencies)

    return comp_freq_dict


def corpus_subject_object_freq(corp, genders, pickle_filepath=None):
    """
    Takes in a Corpus of novels and genders to look for.
    Returns a dictionary of dictionaries, one for each gender
    Each dictionary maps each Document in the corpus to the proportion of the pronouns
    of the specified gender in that novel that are subject pronouns

    :param corp: Corpus object
    :param genders: a list of Gender objects to compare
    :param pickle_filepath: Location to store results; will not write a file if None
    :return: dictionary of results, by document and then by gender.

    >>> from corpus_analysis.corpus import Corpus
    >>> from gender_analysis.analysis.gender_frequency import corpus_subject_object_freq
    >>> from gender_analysis.common import MALE, FEMALE, BINARY_GROUP
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH, SMALL_TEST_CORPUS_CSV
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=SMALL_TEST_CORPUS_CSV, ignore_warnings = True)
    >>> pronoun_freqs = corpus_subject_object_freq(corpus, BINARY_GROUP)
    >>> result = pronoun_freqs.popitem()
    >>> result[1][FEMALE]
    {'subj': 0.4872013651877133, 'obj': 0.5127986348122866}
    """
    relative_freq = {}

    for document in corp.documents:
        relative_freq[document] = document_subject_object_freq(document, genders)

    if pickle_filepath:
        for gender in genders:
            gender_path = Path.join(pickle_filepath, gender.label)
            common.store_pickle(relative_freq, gender_path)

    return relative_freq


def document_subject_object_freq(document, genders):
    """
    Takes in a Document and genders to look for.
    Returns a dictionary of dictionaries, one for each gender
    Each dictionary maps each gender to the proportion of the pronouns
    of the specified gender in that novel that are subject pronouns

    :param document: Document object
    :param genders: a list of Gender objects to compare
    :return: dictionary of results, by gender.
    """
    freq = {}
    relative_freq = {}

    for gender in genders:
        freq[gender] = {}
        relative_freq[gender] = {}
        freq[gender]['subj'] = 0
        freq[gender]['obj'] = 0
        gender_subjects = gender.subj.copy()
        gender_objects = gender.obj.copy()
        for _ in range(0, len(gender_subjects)):
            subject_pronoun = gender_subjects.pop()
            freq[gender]['subj'] += document.get_word_freq(subject_pronoun)
        for _ in range(0, len(gender_objects)):
            object_pronoun = gender_objects.pop()
            freq[gender]['obj'] += document.get_word_freq(object_pronoun)

        comp_freq = get_comparative_word_freq(freq[gender])
        relative_freq[gender]['subj'] = comp_freq['subj']
        relative_freq[gender]['obj'] = comp_freq['obj']

    return relative_freq


def corpus_sub_pronouns_gender_comparison(corp, gender_to_include, genders_to_exclude):
    """
    Takes in a Corpus of novels and a gender.
    The gender determines whether the male frequency or female frequency will be returned.

    Returns a dictionary of each novel in the Corpus mapped to the portion of the subject
    pronouns in the book that are of the specified gender.

    :param corp: Corpus object
    :param gender_to_include: a Gender object to look for
    :param genders_to_exclude: A list of Gender objects to compare against
    :return: dictionary

    >>> from corpus_analysis.corpus import Corpus
    >>> from gender_analysis.common import MALE, FEMALE
    >>> from gender_analysis.analysis.gender_frequency import corpus_sub_pronouns_gender_comparison
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH, SMALL_TEST_CORPUS_CSV
    >>> c = Corpus(TEST_CORPUS_PATH, csv_path=SMALL_TEST_CORPUS_CSV, ignore_warnings = True)
    >>>
    """
    # TODO: Finish this test.
    # TODO: Work on this pickling. I am not convinced that this is a place for pickling.

    # try:
    #     relative_freq = common.load_pickle(pickle_filepath)
    # except IOError:
    #     pass

    relative_freq_sub = {}

    for doc in corp.documents:
        relative_freq_sub[doc] = {}
        relative_freq_sub[doc] = doc_subject_pronouns_gender_comparison(doc, gender_to_include,
                                                                        genders_to_exclude)

        # TODO: Do we need this pickling? None of this stuff seems too intensive, it's all counts.
        # if pickle_filepath_male and pickle_filepath_female:
        #     common.store_pickle(relative_freq_female_sub,
        #                         pickle_filepath_female)
        #     common.store_pickle(relative_freq_male_sub, pickle_filepath_male)

    return relative_freq_sub


def doc_subject_pronouns_gender_comparison(doc, gender_to_include, genders_to_exclude):

    """
    Takes in a document, a gender to look for, and a list of genders to count against.

    Returns the portion of the subject pronouns in the specified doc that are of the included gender

    :param doc: Document object
    :param gender_to_include: a Gender object to look for
    :param genders_to_exclude: A list of Gender objects to compare against
    :return: A float

    >>> from corpus_analysis.corpus import Corpus
    >>> from gender_analysis.common import MALE, FEMALE
    >>> from gender_analysis.analysis.gender_frequency import doc_subject_pronouns_gender_comparison
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH, SMALL_TEST_CORPUS_CSV
    >>> corpus = Corpus(TEST_CORPUS_PATH, csv_path=SMALL_TEST_CORPUS_CSV, ignore_warnings = True)
    >>> emma = corpus.get_document('title', 'Emma')
    >>> doc_subject_pronouns_gender_comparison(emma, FEMALE, [MALE])
    0.563082347015865
    """
    subject_numerator = 0
    subjects_to_include = gender_to_include.subj.copy()

    for _ in range(0, len(subjects_to_include)):
        subject_to_include = subjects_to_include.pop()
        subject_numerator += doc.get_word_freq(subject_to_include)

    subject_denomenator = subject_numerator

    for gender in genders_to_exclude:
        subjects_to_exclude = gender.subj.copy()
        for _ in range(0, len(subjects_to_exclude)):
            subject_to_exclude = subjects_to_exclude.pop()
            subject_denomenator += doc.get_word_freq(subject_to_exclude)

    relative_freq_sub = subject_numerator / subject_denomenator

    return relative_freq_sub


def freq_by_author_gender(novel_dict, genders):
    """
    Takes in a dictionary of novel objects mapped to relative frequencies:
    (from **corpus_pronoun_freq**,**subject_vs_object_freqs**,
    or **subject_pronouns_gender_comparison**).

    Returns a dictionary with frequencies binned by author gender and then document.

    list names key:

    :param novel_dict: dictionary
    :param genders: a list of Gender objects
    :return: dictionary

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH
    >>> from gender_analysis.common import MALE, FEMALE
    >>> novel_metadata = {'filename': 'bronte_wildfell.txt', 'author_gender': 'female',
    ...    'filepath': Path(TEST_CORPUS_PATH, 'bronte_wildfell.txt')}
    >>> bronte = document.Document(novel_metadata)
    >>> novel_metadata = {'filename': 'adams_fighting.txt', 'author_gender': 'male',
    ...                   'filepath': Path(TEST_CORPUS_PATH, 'adams_fighting.txt')}
    >>> fighting = document.Document(novel_metadata)
    >>> dictionary = {fighting:0.3, bronte:0.6}
    >>> freq_by_author_gender(dictionary, [MALE, FEMALE])
    {'Male Authors': [0.3], 'Female Authors': [0.6]}

    """

    data = {}

    for gender in genders:
        label = gender.label + " Authors"
        data[label] = []
        for k, val in novel_dict.items():
            author_gender = getattr(k, 'author_gender', None)
            if author_gender == gender.label or author_gender == gender.label.lower():
                data[label].append(val)

    return data


def freq_by_date(doc_dict, time_frame, bin_size):
    """
    Takes in a dictionary of Document objects mapped to relative pronoun frequencies, and
    returns a dictionary with frequencies binned by decades into lists
    List name is mapped to the list of frequencies

    list names key:
    date_to_1810 - publication dates before and not including 1810
    date_x_to_y (by decade) - publication dates from x to y
    Example: date_1810_to_1819 - publication dates from 1810 to 1819
    date_1900_on - publication dates in 1900 and onward

    :param doc_dict: dictionary
    :param time_frame: tuple (int start year, int end year) for the range of dates to return freqs
    :param bin_size: int for the number of years represented in each list of frequencies
    :return: dictionary {bin_start_year:[frequencies for documents in this bin of years]

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH
    >>> from gender_analysis.analysis.gender_frequency import freq_by_date
    >>> novel_metadata = {'date': '1818', 'filename': 'austen_persuasion.txt',
    ...                   'filepath': Path(TEST_CORPUS_PATH, 'austen_persuasion.txt')}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata = {'date': '1900', 'filename': 'hawthorne_scarlet.txt',
    ...                   'filepath': Path(TEST_CORPUS_PATH, 'hawthorne_scarlet.txt')}
    >>> scarlet = document.Document(novel_metadata)
    >>> dictionary = {scarlet:0.5, austen:0.3}
    >>> freq_by_date(dictionary, (1800, 1900), 20)
    {1800: [0.3], 1820: [], 1840: [], 1860: [], 1880: [], 1900: [0.5]}

    """

    data = {}
    for bin_start_year in range(time_frame[0], time_frame[1] + bin_size, bin_size):
        data[bin_start_year] = []

    for k, val in doc_dict.items():
        date = getattr(k, 'date', None)
        if date is None:
            continue
        bin_year = ((date - time_frame[0]) // bin_size) * bin_size + time_frame[0]
        if bin_year not in data.keys():
            continue
        data[bin_year].append(val)

    return data


def freq_by_location(novel_dict):
    """
    Takes in a dictionary of novel objects mapped to relative frequencies.
    Returns a dictionary with frequencies binned by publication location into lists

    list names key:

    - *location_UK* - published in the United Kingdom
    - *location_US* - published in the US
    - *location_other* - published somewhere other than the US and England

    :param novel_dict: dictionary
    :return: dictionary

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH
    >>> from gender_analysis.analysis.gender_frequency import freq_by_location
    >>> novel_metadata = {'country_publication': 'United Kingdom',
    ...                   'filename': 'austen_persuasion.txt',
    ...                   'filepath': Path(TEST_CORPUS_PATH, 'austen_persuasion.txt')}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata2 = {'country_publication': 'United States',
    ...                    'filename':'hawthorne_scarlet.txt',
    ...                    'filepath': Path(TEST_CORPUS_PATH, 'hawthorne_scarlet.txt')}
    >>> scarlet = document.Document(novel_metadata2)
    >>> dictionary = {scarlet:0.5, austen:0.3}
    >>> freq_by_location(dictionary)
    {'United States': [0.5], 'United Kingdom': [0.3]}
    """
    data = {}

    for k, val in novel_dict.items():
        location = getattr(k, 'country_publication', None)
        if location is None:
            continue

        if location not in data:
            data[location] = []
        data[location].append(val)

    return data


def get_mean(data_dict):
    """
    Takes in a dictionary matching some object to lists and returns a dictionary of the
        original keys mapped to the mean of the lists

    :param data_dict: dictionary matching some object to lists
    :return: dictionary with original key mapped to an average of the input list

    >>> d = {'fives': [5,5,5], 'halfway': [0,1], 'nothing': [0]}
    >>> get_mean(d)
    {'fives': 5.0, 'halfway': 0.5, 'nothing': 0.0}
    """
    mean_dict = {}
    for k, val in data_dict.items():
        try:
            mean_dict[k] = np.mean(val)
        except ZeroDivisionError:
            mean_dict[k + "*"] = 0.5
    return mean_dict


def sort_every_year(frequency_dict):
    """
    Takes in a dictionary of documents mapped to pronoun frequencies and returns a dictionary of
    years mapped to lists of pronoun frequencies

    :param frequency_dict: dictionary of documents mapped to pronoun frequencies
    :return: dictionary of years mapped to lists of pronoun frequencies

    >>> from corpus_analysis import document
    >>> from pathlib import Path
    >>> from gender_analysis import common
    >>> from corpus_analysis.testing.common import TEST_CORPUS_PATH
    >>> novel_metadata = {'date': '1818', 'filename': 'austen_persuasion.txt',
    ...                   'filepath': Path(TEST_CORPUS_PATH, 'austen_persuasion.txt')}
    >>> austen = document.Document(novel_metadata)
    >>> novel_metadata = {'date': '1900', 'filename': 'hawthorne_scarlet.txt',
    ...                   'filepath': Path(TEST_CORPUS_PATH, 'hawthorne_scarlet.txt')}
    >>> scarlet = document.Document(novel_metadata)
    >>> d = {scarlet:0.5, austen:0.3}
    >>> sorted_years = sort_every_year(d)
    >>> print(sorted_years)
    {1900: [0.5], 1818: [0.3]}

    """

    every_year_dict = {}
    for key, _ in frequency_dict.items():
        # TODO: check if key (document?) has date attribute
        frequency_list = [frequency_dict[key]]

        if key.date not in every_year_dict.keys():
            every_year_dict[key.date] = frequency_list

        elif key.date in every_year_dict.keys():
            every_year_dict[key.date].append(frequency_dict[key])

    return every_year_dict
