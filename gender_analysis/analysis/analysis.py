"""
This file is intended for individual analyses of the gender_analysis project
"""

import nltk
import numpy as np
import matplotlib.pyplot as plt
from more_itertools import windowed
import collections
from statistics import median
from nltk.corpus import stopwords
import unittest
from operator import itemgetter
from gender_analysis.corpus import Corpus
import seaborn as sns

from gender_analysis.analysis.dunning import dunn_individual_word, dunn_individual_word_by_corpus

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

palette = "colorblind"
style_name = "white"
style_list = {'axes.edgecolor': '.6', 'grid.color': '.9', 'axes.grid': 'True',
                           'font.family': 'serif'}
sns.set_color_codes(palette)
sns.set_style(style_name,style_list)


def get_count_words(document, words):
    """
    Takes in document, a Document object, and words, a list of words to be counted.
    Returns a dictionary where the keys are the elements of 'words' list
    and the values are the numbers of occurrences of the elements in the document.
    N.B.: Not case-sensitive.
    >>> from gender_analysis import document
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ...                   'corpus_name': 'document_test_files', 'date': '1850',
    ...                   'filename': 'test_text_2.txt'}
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
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ...                   'corpus_name': 'sample_novels', 'date': '1900',
    ...                   'filename': 'hawthorne_scarlet.txt'}
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

    >>> get_counts_by_pos(collections.Counter({'baked':1,'chair':3,'swimming':4}))
    {'VBN': Counter({'baked': 1}), 'NN': Counter({'chair': 3}), 'VBG': Counter({'swimming': 4})}
    >>> get_counts_by_pos(collections.Counter({'is':10,'usually':7,'quietly':42}))
    {'RB': Counter({'quietly': 42, 'usually': 7})}

    :param freqs: Counter object of words mapped to their word count
    :return: dictionary with key as part of speech, value as Counter object of words (of that
    part of speech) mapped to their word count
    """

    sorted_words = {}
    # for each word in the counter
    for word in freqs.keys():
        # filter out if in nltk's list of stop words, e.g. is, the
        if word not in stop_words:
            # get its part of speech tag from nltk's pos_tag function
            tag = nltk.pos_tag([word])[0][1]
            # add that word to the counter object in the relevant dict entry
            if tag not in sorted_words.keys():
                sorted_words[tag] = collections.Counter({word:freqs[word]})
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
    loops = c//10 + 1

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


def dunning_total(m_corpus, f_corpus):
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
         >>> c = Corpus('sample_novels')
         >>> m_corpus = c.filter_by_gender('male')
         >>> f_corpus = c.filter_by_gender('female')
         >>> result = dunning_total(m_corpus, f_corpus)
         >>> print(result[0:10])
         [('she', (-12292.762338290115, 29042, 45509)), ('her', (-11800.614222528242, 37517, 53463)), ('jo', (-3268.940103481869, 1, 1835)), ('carlyle', (-2743.3204833572668, 3, 1555)), ('mrs', (-2703.877430262923, 3437, 6786)), ('amy', (-2221.449213948045, 36, 1408)), ('laurie', (-1925.9408323278521, 2, 1091)), ('adeline', (-1896.0496657740907, 13, 1131)), ('alessandro', (-1804.1775207769476, 3, 1029)), ('mr', (-1772.0584351647658, 7900, 10220))]
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


def instance_dist(document, word):
    """
    Takes in a particular word, returns a list of distances between each instance of that word in
    the document.
    >>> from gender_analysis import document
    >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
    ...                   'corpus_name': 'document_test_files', 'date': '1966',
    ...                   'filename': 'test_text_3.txt'}
    >>> scarlett = document.Document(document_metadata)
    >>> instance_dist(scarlett, "her")
    [6, 5, 6, 7, 7]

    :param: document: Document to analyze
    :param: word: str
    :return: list of distances between consecutive instances of word

    """
    return words_instance_dist(document, [word])


def words_instance_dist(document, words):
    """
        Takes in a document and list of words (e.g. gendered pronouns), returns a list of distances
        between each instance of one of the words in that document
        >>> from gender_analysis import document
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
        ...                   'corpus_name': 'document_test_files', 'date': '1966',
        ...                   'filename': 'test_text_4.txt'}
        >>> scarlett = document.Document(document_metadata)
        >>> words_instance_dist(scarlett, ["his", "him", "he", "himself"])
        [6, 5, 6, 6, 7]

        :param: document: Document
        :param: words: list of strings
        :return: list of distances between instances of any word in words
    """
    text = document.get_tokenized_text()
    output = []
    count = 0
    start = False

    for token in text:
        token = token.lower()
        if not start:
            if token in words:
                start = True
        else:
            count += 1
            if token in words:
                output.append(count)
                count = 0
    return output


def male_instance_dist(document):
    """
        Takes in a document, returns a list of distances between each instance of a female pronoun
        in that document.
       >>> from gender_analysis import document
       >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
       ...                   'corpus_name': 'document_test_files', 'date': '1966',
       ...                   'filename': 'test_text_5.txt'}
       >>> scarlett = document.Document(document_metadata)
       >>> male_instance_dist(scarlett)
       [6, 5, 6, 6, 7]

       :param: document
       :return: list of distances between instances of gendered word
    """
    return words_instance_dist(document, ["his", "him", "he", "himself"])


def female_instance_dist(document):
    """
        Takes in a document, returns a list of distances between each instance of a female pronoun
        in that document.
       >>> from gender_analysis import document
       >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
       ...                   'corpus_name': 'document_test_files', 'date': '1966',
       ...                   'filename': 'test_text_6.txt'}
       >>> scarlett = document.Document(document_metadata)
       >>> female_instance_dist(scarlett)
       [6, 5, 6, 6, 7]

       :param: document
       :return: list of distances between instances of gendered word
    """
    return words_instance_dist(document, ["her", "hers", "she", "herself"])


def find_gender_adj(document, female):
    """
        Takes in a document and boolean indicating gender, returns a dictionary of adjectives that
        appear within a window of 5 words around each pronoun
        >>> from gender_analysis import document
        >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
        ...                   'corpus_name': 'document_test_files', 'date': '1966',
        ...                   'filename': 'test_text_7.txt'}
        >>> scarlett = document.Document(document_metadata)
        >>> find_gender_adj(scarlett, False)
        {'handsome': 3, 'sad': 1}

        :param: document: Document
        :param: female: boolean indicating whether to search for female adjectives (true) or
        male adj (false)
        :return: dictionary of adjectives that appear around male pronouns and the number of
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
       >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
       ...                   'corpus_name': 'document_test_files', 'date': '1966',
       ...                   'filename': 'test_text_8.txt'}
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
       >>> document_metadata = {'author': 'Hawthorne, Nathaniel', 'title': 'Scarlet Letter',
       ...                   'corpus_name': 'document_test_files', 'date': '1966',
       ...                   'filename': 'test_text_9.txt'}
       >>> scarlett = document.Document(document_metadata)
       >>> find_female_adj(scarlett)
       {'beautiful': 3, 'sad': 1}

       :param:document
       :return: dictionary of adjectives that appear around female pronouns and the number of
       occurrences

       """
    return find_gender_adj(document, True)


def process_medians(helst, shelst, authlst):
    """
    >>> medians_he = [12, 130, 0, 12, 314, 18, 15, 12, 123]
    >>> medians_she = [123, 52, 12, 345, 0,  13, 214, 12, 23]
    >>> books = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    >>> process_medians(helst=medians_he, shelst=medians_she, authlst=books)
    {'he': [0, 2.5, 0, 1.3846153846153846, 0, 1.0, 5.3478260869565215], 'she': [10.25, 0, 28.75, 0, 14.266666666666667, 0, 0], 'book': ['a', 'b', 'd', 'f', 'g', 'h', 'i']}

    :param helst:
    :param shelst:
    :param authlst:
    :return: a dictionary sorted as so {
                                        "he":[ratio of he to she if >= 1, else 0], "she":[ratio of she to he if > 1, else 0] "book":[lst of book authors]
                                       }
    """
    d = {"he": [], "she": [], "book": []}
    for num in range(len(helst)):
        if helst[num] > 0 and shelst[num] > 0:
            res = helst[num] - shelst[num]
            if res >= 0:
                d["he"].append(helst[num] / shelst[num])
                d["she"].append(0)
                d["book"].append(authlst[num])
            else:
                d["he"].append(0)
                d["she"].append(shelst[num] / helst[num])
                d["book"].append(authlst[num])
        else:
            if helst == 0:
                print("ERR: no MALE values: " + authlst[num])
            if shelst == 0:
                print("ERR: no FEMALE values: " + authlst[num])
    return d


def bubble_sort_across_lists(dictionary):
    """
    >>> d = {'he': [0, 2.5, 0, 1.3846153846153846, 0, 1.0, 5.3478260869565215],
    ...     'she': [10.25, 0, 28.75, 0, 14.266666666666667, 0, 0],
    ...     'book': ['a', 'b', 'd', 'f', 'g', 'h', 'i']}
    >>> bubble_sort_across_lists(d)
    {'he': [5.3478260869565215, 2.5, 1.3846153846153846, 1.0, 0, 0, 0], 'she': [0, 0, 0, 0, 10.25, 14.266666666666667, 28.75], 'book': ['i', 'b', 'f', 'h', 'a', 'g', 'd']}

    :param dictionary: containing 3 different list values.
    Note: dictionary keys MUST contain arguments 'he', 'she', and 'book'
    :return dictionary sorted across all three lists in a specific method:
    1) Descending order of 'he' values
    2) Ascending order of 'she' values
    3) Corresponding values of 'book' values
    """
    lst1 = dictionary['he']
    lst2 = dictionary['she']
    lst3 = dictionary['book']
    r = range(len(lst1) - 1)
    p = True

    # sort by lst1 descending
    for j in r:
        for i in r:
            if lst1[i] < lst1[i + 1]:
                # manipulating lst 1
                temp1 = lst1[i]
                lst1[i] = lst1[i + 1]
                lst1[i + 1] = temp1
                # manipulating lst 2
                temp2 = lst2[i]
                lst2[i] = lst2[i + 1]
                lst2[i + 1] = temp2
                # manipulating lst of authors
                temp3 = lst3[i]
                lst3[i] = lst3[i + 1]
                lst3[i + 1] = temp3
                p = False
        if p:
            break
        else:
            p = True

    # sort by lst2 ascending
    for j in r:
        for i in r:
            if lst2[i] > lst2[i + 1]:
                # manipulating lst 1
                temp1 = lst1[i]
                lst1[i] = lst1[i + 1]
                lst1[i + 1] = temp1
                # manipulating lst 2
                temp2 = lst2[i]
                lst2[i] = lst2[i + 1]
                lst2[i + 1] = temp2
                # manipulating lst of authors
                temp3 = lst3[i]
                lst3[i] = lst3[i + 1]
                lst3[i + 1] = temp3
                p = False
        if p:
            break
        else:
            p = True
    d = {}
    d['he'] = lst1
    d['she'] = lst2
    d['book'] = lst3
    return d


def instance_stats(book, medians1, medians2, title):
    """
    :param book:
    :param medians1:
    :param medians2:
    :param title: str, desired name of file
    :return: None, file written to visualizations folder depicting the ratio of two values given
    as a bar graph
    """
    fig, ax = plt.subplots()
    plt.ylim(0, 50)

    index = np.arange(len(book))
    bar_width = .7
    opacity = 0.4

    medians_she = tuple(medians2)
    medians_he = tuple(medians1)
    book = tuple(book)

    rects1 = ax.bar(index, medians_he, bar_width, alpha=opacity, color='b', label='Male to Female')
    rects2 = ax.bar(index, medians_she, bar_width, alpha=opacity, color='r', label='Female to Male')

    ax.set_xlabel('Book')
    ax.set_ylabel('Ratio of Median Values')
    ax.set_title(
        'MtF or FtM Ratio of Median Distance of Gendered Instances by Author')
    ax.set_xticks(index)
    plt.xticks(fontsize=8, rotation=90)
    ax.set_xticklabels(book)
    ax.set_yscale("symlog")

    ax.legend()

    fig.tight_layout()
    filepng = "visualizations/" + title + ".png"
    filepdf = "visualizations/" + title + ".pdf"
    plt.savefig(filepng, bbox_inches='tight')
    plt.savefig(filepdf, bbox_inches='tight')


def run_dist_inst(corpus):
    """
    Runs a program that uses the instance distance analysis on all documents existing in a given
    corpus, and outputs the data as graphs
    :param corpus:
    :return:
    """
    documents = corpus.documents
    c = len(documents)
    loops = c//10 + 1

    num = 0

    while num < loops:
        medians_he = []
        medians_she = []
        books = []
        for document in documents[num * 10: min(c, num * 10 + 9)]:
            result_he = instance_dist(document, "he")
            result_she = instance_dist(document, "she")
            try:
                medians_he.append(median(result_he))
            except:
                medians_he.append(0)
            try:
                medians_she.append(median(result_she))
            except:
                medians_she.append(0)
            title = hasattr(document, 'title')
            author = hasattr(document, 'author')
            if title and author:
                books.append(document.title[0:20] + "\n" + document.author)
            elif title:
                books.append(document.title[0:20])
            else:
                books.append(document.filename[0:20])
        d = process_medians(helst=medians_he, shelst=medians_she, authlst=books)
        d = bubble_sort_across_lists(d)
        instance_stats(d["book"], d["he"], d["she"], "inst_dist" + str(num))
        num += 1


class Test(unittest.TestCase):
    def test_dunning_total(self):
        c = Corpus('sample_novels')
        m_corpus = c.filter_by_gender('male')
        f_corpus = c.filter_by_gender('female')
        results = dunning_total(m_corpus, f_corpus)
        print(results[10::])
